# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error
import time
import psutil

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="Gerado por IA"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

#----------------------------------------------------------------------------

# Classe para renderização de imagens
class Renderer:
    # Inicializa o objeto Renderer
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    # Gerencia o processo de renderização
    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args) # Chama a Implementação da Renderização
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    # Carrega e armazena em cache as redes neurais
    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            #print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                #print('Done.')
            except:
                data = CapturedException()
                #print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                #print(data[key].init_args)
                #print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    # Gerencia a fixação de memória para tensores para otimizar a transferência de dados entre CPU e GPU
    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    # Método para mover dados para o dispositivo
    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    # Método para mover dados para a cpu    
    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    # Método utilitário para desativar a temporização para certas operações
    def _ignore_timing(self):
        self._is_timing = False

    # Aplica um mapa de cores ao tensor fornecido, usado para fins de visualização
    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    # Inicializa e configura a rede neural - executada somente quando o gradio é inicializado
    def init_network(self, res,
        pkl             = None,     # Caminho ou identificador para o arquivo de modelo pre-treinado
        w0_seed         = 0,        # Parâmetros relacionados a códigos latentes iniciais
        w_load          = None,     # Parâmetros relacionados a códigos latentes iniciais
        w_plus          = True,     # Um booleano para determinar se deve usar o esquema de códigos latentes "w plus"
        noise_mode      = 'const',  # Parâmetro de controle para a geração de imagem
        trunc_psi       = 0.7,      # Parâmetro de controle para a geração de imagem
        trunc_cutoff    = None,     # Parâmetro de controle para a geração de imagem
        input_transform = None,     # Transformação de entrada
        lr              = 0.002,    # Taxa de aprendizado para ajustes na rede
        **kwargs
        ):

        # Armazena o caminho ou identificador do modelo
        self.pkl = pkl 

        # Carrega a rede neural como um módulo PyTorch
        G = self.get_network(pkl, 'G_ema') 

        # Atribui a rede carregada a uma variável de instância
        self.G = G 
        
        # Armazena detalhes da imagem e da rede no objeto res
        res.img_resolution = G.img_resolution 
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Configura a Transformação de Entrada
        if res.has_input_transform:
            
            # Retorna uma matriz 2-D com uns na diagonal e zeros em outros lugares
            m = np.eye(3)
            try:
                # Verifica se a rede possui uma transformação de entrada específica e, se sim, tenta aplicá-la usando matrizes de transformação
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()

            # Copia os dados da matriz m para o tensor de transformação de entrada
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Inicializa o contador de iterações e variáveis de controle
        self.num_iteracoes = 0
        self.num_tot_iteracoes_trad = 0
        self.num_tot_iteracoes_prop = 0
        self.dist_ini_pi_ti = None
        self.ITERACAO_MAX = 2
        self.distancias_anteriores = None
        self.distancia_invalida = False
        self.exibiu_log = False
        res.resetar = False
        self.start_time = 0
        self.gpu_memory_allocated_before = 0
        self.gpu_memory_reserved_before = 0

        # Geração de Latentes Aleatórios
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Gera um vetor latente z aleatório
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Cria um tensor de rótulos
                # G.c_dim: é a dimensão do espaço de rótulos condicionais
                # self._device: é o dispositivo onde o tensor deve ser alocado
                # Inicializa como um tensor de zeros
            label = torch.zeros([1, G.c_dim], device=self._device)
            
            # Execução da Rede de Mapeamento 
                # G.mapping: rede de mapeamento que transforma um vetor latente inicial z em um vetor latente intermediário w
                # label: rótulos condicionais
                # truncation_psi e truncation_cutoff: parâmetros da StyleGAN que controlam a "truncagem" do espaço latente
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        # Armazena o estado inicial do vetor latente w
            # criar uma nova visão do tensor w que não requer gradiente
            # detach(): não influenciará no cálculo do gradiente durante o processo de otimização
            # clone(): cria uma cópia do tensor que é completamente nova e independente do original
                # alterações futuras em self.w_inicial não afetarão o tensor w original
        self.w_inicial = w.detach().clone()

        # Armazena o estado inicial do vetor latente w
        self.w0 = w.detach().clone()
        
        # w_plus é um booleano que indica se deve usar o espaço latente estendido "W+" em vez do espaço "W" regular
        self.w_plus = w_plus
        
        if w_plus:
            self.w = w.detach()
        else:
            # atualiza o estado de self.w com
                # todos elementos na primeira dimensão de w (Batch Size: 1 gera uma imagem por vez)
                # seleciona o primeiro elemento da segunda dimensão (número de camadas -> apenas a primeira camada)
                # todos elementos na primeira dimensão de w (todos os componentes do vetor latente)
            self.w = w[:, 0, :].detach()
        
        # Calcula o gradiente durante a backpropagation para todas as operações no tensor w
            # Permite ajustar os valores de w para minimizar a função de perda durante a otimização
        self.w.requires_grad = True
        
        # torch.optim.Adam: chamada ao otimizador Adam 
            # (uma variante do método de descida de gradiente estocástico que ajusta cada parâmetro com taxas de aprendizado individualmente adaptáveis)
            # [self.w]: lista de tensores com um tensor
            # lr: taxa de aprendizado (learning rate) em float - influencia a velocidade e a qualidade da convergência do processo de otimização
                # Uma taxa de aprendizado muito alta pode fazer com que o otimizador "pule" o mínimo, 
                # enquanto uma taxa muito baixa pode fazer com que o treinamento seja muito lento ou fique preso em mínimos locais
        self.w_optim = torch.optim.Adam([self.w], lr=lr)

        # Inicializa variáveis que serão usadas para referenciar características ou pontos específicos durante a manipulação de imagens
        self.feat_refs = None
        self.points0_pt = None
        
    # Atualiza a taxa de aprendizado do otimizador, útil para ajustes finos durante o processo de renderização
    def update_lr(self, lr):
        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        #print(f'Rebuild optimizer with lr: {lr}')
        #print('    Remain feat_refs and points0_pt')

    # Implementa a lógica de renderização
    def _render_drag_impl(self, res,    # res recebe global_state['generator_params']
        points          = [],           # Coordenadas dos pontos de manipulação
        targets         = [],           # Coordenadas dos pontos alvo
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 36,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        **kwargs
    ):
        G = self.G
        
        # ws: vetor latente atual
        ws = self.w

        if ws.dim() == 2:
            # Ajuste do formato de ws para corresponder a quantidade de camadas da rede
                # unsqueeze(1): adiciona uma dimensão de tamanho 1 ao tensor ws na posição 1. 
                    # se ws tiver uma forma digamos [x, y] após ws.unsqueeze(1) ele vai virar [x, 1, y]
                # repeat(1,6,1): repetir o tensor ao longo de cada dimensão especificada
                    # A primeira dimensão não é repetida (1 vez).
                    # A segunda dimensão (a que acabamos de adicionar com unsqueeze) é repetida 6 vezes.
                    # A terceira dimensão não é repetida (1 vez).
            ws = ws.unsqueeze(1).repeat(1,6,1)
        
        # O tensor ws recebe uma concatenação de uma sequência de tensores 
            # [ws[:,:6,:], self.w0[:,6:,:]]: lista de tensores a serem concatenados
                # ws[:,:6,:]: fatia de ws ==> pega as primeiras 6 dimensões da segunda dimensão
                # self.w0[:,6:,:]: fatia de self.w0 ==> pega tudo a partir da sétima camada da segunda dimensão
            # dim=1: dimensão ao longo da qual a concatenação deve ocorrer (segunda dimensão)
        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        
        # Reset de referências
        if reset:
            self.feat_refs = None  # referências de características
            self.points0_pt = None # referências de pontos
            
            #print(70 * "*" + "Inicializa a contagem de tempo e de memória")
            self.start_time = time.time() ## INICIAR O TIMER AQUI!
            self.gpu_memory_allocated_before = torch.cuda.memory_allocated()
            self.gpu_memory_reserved_before = torch.cuda.memory_reserved()    

        if res.resetar:
            #print("\n\nResetei!")
            self.num_iteracoes = 0 # zera o contador de iterações
            self.num_tot_iteracoes_trad = 0
            self.num_tot_iteracoes_prop = 0
            self.dist_ini_pi_ti = None
            self.w_inicial = self.w.detach().clone() # carrega atualiza o w_inicial com o w atual
            self.distancias_anteriores = None
            self.exibiu_log = False
            self.distancia_invalida = False
            res.resetar = False

        self.points = points

        # Cria um tensor label onde todos elementos são zero
            # [1, G.c_dim]: fortmato do tensor a ser criado
                # 1 é a dimensão do lote - uma imagem por vez
        label = torch.zeros([1, G.c_dim], device=self._device)
        
        # Executa a rede de síntese
            # G: rede geradora
            # img: imagem gerada pela rede
            # feat: mapas de features
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

        # resoluções da altura e largura: imagem quadrada
        h, w = G.img_resolution, G.img_resolution

        if is_drag:
            
            # Cria duas grades 1D de valores lineare
                # torch.linspace(start, end, steps): cria um tensor de 1 dimensão contendo steps números igualmente espaçados entre start e end
                # o tensor terá um total de steps elementos
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            
            # xx e yy são grades 2D onde xx[i, j] e yy[i, j] representam as coordenadas X e Y do ponto na posição (i, j) na imagem.
            # torch.meshgrid(...): pega dois tensores 1D e produz dois tensores 2D que correspondem a todas as combinações de X e Y
            # usado para criar uma grade de coordenadas
            # xx: Para cada posição na grade, xx contém o valor da coordenada X correspondente
            # yy: Para cada posição na grade, yy contém o valor da coordenada Y correspondente
            xx, yy = torch.meshgrid(X, Y)
            
            # Redimensiona as features do sexto bloco da Stylegan2 para terem a mesma dimensão da imagem original
                # Geralmente h = 512 e w = 512
            feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
            
            if self.feat_refs is None:

                # Redimensiona as Características  do bloco feature_idx = 5 (sexto bloco) da Stylegan2
                    # Características iniciais e  garante que elas não rastreiem operações para cálculo do gradiente, tornando-as constantes
                self.feat0_resize = F.interpolate(feat[feature_idx].detach(), [h, w], mode='bilinear')
                
                # lista com as características específicas dos pontos de interesse
                self.feat_refs = [] 
                
                # Itera sobre cada ponto fornecido em points (coord dos pontos de manipulação)
                for point in points:
                    # Coordenadas arredondadas y (altura) e x (largura) do ponto de manipulação
                    py, px = round(point[0]), round(point[1])
                    
                    # Adiciona as características na localização especificada (py, px) à lista self.feat_refs
                    self.feat_refs.append(self.feat0_resize[:,:,py,px])
                
                # Configuração de self.points0_pt:
                    # torch.Tensor(points): Converte a lista de pontos em um tensor PyTorch
                    # .unsqueeze(0): Adiciona uma dimensão extra no início do tensor, transformando-o 
                        # de uma matriz de forma [N, 2] para [1, N, 2], onde N é o número de pontos.
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2

            #----------------------------------------------------------------------------
            # Rastreamento de pontos com correspondência de recursos
            #----------------------------------------------------------------------------

            with torch.no_grad(): # Desabilitando o Cálculo do Gradiente
                
                # Itera sobre cada ponto na lista points. j é o índice e point são as coordenadas (y, x) do ponto.
                for j, point in enumerate(points):
                    # Calcula um raio r baseado em r2 que será usado para definir uma região quadrada ao redor de cada ponto
                    r = round(r2 / 512 * h)
                    
                    # Limites superior, inferior, esquerdo e direito da região ao redor de cada ponto, usando o raio r
                        # Garantem que os limites não ultrapassem as bordas da imagem, usando max e min para restringir os valores
                    up = max(point[0] - r, 0)
                    down = min(point[0] + r + 1, h)
                    left = max(point[1] - r, 0)
                    right = min(point[1] + r + 1, w)
                    
                    # Extrai um patch de características da imagem redimensionada feat_resize que corresponde à região ao redor do ponto atual
                    feat_patch = feat_resize[:,:,up:down,left:right]

                    # Calcula a norma L2 (ou norma Euclidiana) da diferença entre o patch de características extraído e a referência 
                    # de características armazenada para o ponto atual. Isso efetivamente mede a distância entre as características do 
                    # patch atual e as características originais, fornecendo uma medida de similaridade.
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                    
                    # Encontra a posição dentro do patch de características que tem a menor distância (ou maior similaridade) às características de referência
                    _, idx = torch.min(L2.view(1,-1), -1)

                    # Calcula a largura da região ao redor do ponto
                    width = right - left

                    # Calcula as novas coordenadas do ponto baseando-se no índice de menor distância encontrado. Isso efetivamente 
                    # move o ponto para a posição dentro da região que mais se assemelha às características originais
                    point = [idx.item() // width + up, idx.item() % width + left]
                    
                    # Atualiza a lista de pontos com a nova posição do ponto atual
                    points[j] = point

            # Atualiza o objeto de resultado res com as novas posições dos pontos
            res.points = [[point[0], point[1]] for point in points]

            #----------------------------------------------------------------------------
            # Supervisão de movimentos - primeiro termo
            #----------------------------------------------------------------------------
            
            # Variável para acumular a perda de movimento total - Perda associada ao movimento dos pontos em relação a seus alvos
            loss_motion = 0
            
            # Variável booleana que determinará se o processo de ajuste deve continuar ou parar
            res.stop = True

            if self.distancias_anteriores is None:
                # Se for a primeira iteração, inicialize distancias_anteriores com valores infinitos
                self.distancias_anteriores = torch.tensor([float('inf')] * len(points), dtype=torch.double)
                #print("\nAlimentei o distancias_anteriores com infinito")

            if self.dist_ini_pi_ti is None:
                self.dist_ini_pi_ti = torch.tensor([0] * len(points), dtype=torch.double)

            # Itera sobre cada ponto na lista points. j é o índice e point são as coordenadas (y, x) do ponto.
            for j, point in enumerate(points):

                # Para cada ponto, calcula a diferença vetorial entre o ponto atual e seu alvo correspondente em targets
                # Essa diferença é a "direção" que o ponto precisaria se mover para alcançar o alvo
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                distancia_atual = torch.linalg.norm(direction)
                
                if self.dist_ini_pi_ti[j] == 0:
                    self.dist_ini_pi_ti[j] = distancia_atual

                if torch.round(distancia_atual.double() * 1000) > torch.round(self.distancias_anteriores[j].double() * 1000):
                    #print(80 * "=" + "DISTÂNCIA ENTRE P E T AUMENTOU!")
                    self.distancia_invalida = True
                    self.w_inicial = self.w.detach().clone()
                    self.num_iteracoes = 0
                    res.stop = False
                # Verifica se a norma (comprimento) do vetor direção é maior que um certo limite (o maior entre 2 / 512 * h e 2). 
                # Se for, isso significa que o ponto ainda está significativamente longe do alvo, então res.stop é definido como 
                # False para indicar que o processo de ajuste deve continuar.
                elif torch.linalg.norm(direction) > max(2 / 512 * h, 2):
                    res.stop = False
                
                self.distancias_anteriores[j] = distancia_atual

                #  Se a norma da direção for maior que 1
                if torch.linalg.norm(direction) > 1:
                    
                    # Calcula a distância euclidiana de cada ponto na grade gerada (xx, yy) para o ponto atual (point).
                    # Esta distância é usada para determinar uma região de interesse ao redor do ponto
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    
                    # Identifica índices dentro de uma certa distância (r1) do ponto
                    # Estes índices representam uma região localizada ao redor do ponto onde o ajuste será focado
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))

                    # Normaliza o vetor direção para ter comprimento unitário, evitando problemas relacionados à escala das distâncias
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1

                    # Cria uma grade de mapeamento que será usada para reamostrar as características (feat_resize) na direção do movimento
                    # Isso é feito ajustando as coordenadas da grade com base na direção do movimento
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    # Usa a função grid_sample para reamostrar as características na direção do movimento
                    # cria um novo conjunto de características que representam como as características de imagem seriam se 
                    # o ponto se movesse na direção do alvo
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    
                    # Calcula a perda L1 entre as características reamostradas e as características originais na região 
                    # relevante e a acumula na variável loss_motion
                        # l1_loss: retorna um tensor (de dimensão zero = um escalar)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

            # loss: escalar (tensor de dimensão zero)
            loss = loss_motion
            
            #----------------------------------------------------------------------------
            # Supervisão de movimentos - segundo termo - aplicação da máscara
            #----------------------------------------------------------------------------            
            
            # Se uma máscara foi fornecida ...
            if mask is not None:

                # Verifica se a máscara contém apenas valores binários (0 ou 1)
                if mask.min() == 0 and mask.max() == 1:
                    
                    # A máscara é movida para o dispositivo apropriado (CPU ou GPU) e redimensionada para 
                    # combinar com as dimensões esperadas pelas operações subsequentes
                    # unsqueeze(0) adiciona dimensões extras para tornar a máscara compatível com o formato das características.
                    mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
                    
                    # Calcula a perda L1 entre as características redimensionadas e as características de referência, 
                    # mas apenas nas regiões especificadas pela máscara
                    # Calcula a norma L1 da diferença entre F e F0
                    loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
                    
                    # lambda_mask: hiperparâmetro que equilibra a importância dessa perda específica em relação às outras perda
                    loss += lambda_mask * loss_fix

            # Adiciona uma perda de regularização à perda total, baseada na distância entre o código latente atual ws e um estado 
            # inicial ou referência self.w0. reg é um fator de regularização que controla a importância desta perda.
            # A perda calculada é ponderada pelo fator lambda_mask e adicionada à perda total.
            loss += reg * F.l1_loss(ws, self.w0) # regularização do código latente
            
            # Se o processo não está marcado para parar (res.stop é False), então prossegue com a atualização do código latente
            if not res.stop:

                if self.num_iteracoes == self.ITERACAO_MAX:
                    #print("caí na inicialização 1 do self.w_dif_inicial ") 
                    self.w_dif_inicial = self.w.detach() - self.w_inicial.detach()
                    #print(self.w_dif_inicial)
               
                # Otimização rápida de w
                    # Realizada somente após as primeiras x iterações e apenas para 1 ponto de manipulação.
                if self.num_iteracoes >= self.ITERACAO_MAX and not self.distancia_invalida:
                    with torch.no_grad():
                        #print(80 * "=" + "caí na otimização RÁPIDA ") 
                        self.num_tot_iteracoes_prop += 1
                        self.w = self.w.detach() + self.w_dif_inicial.detach()

                # Otimização original de w 
                else:
                    # print(80 * "=" + "caí na otimização NORMAL ") 
                    # print(f"PERDA: {loss}") 
                    self.num_tot_iteracoes_trad += 1

                    if self.num_iteracoes == 0:
                        #print(80 * "=" + "caí na reinicialização do otimizador Adam  ") 
                        self.w.requires_grad = True
                        self.w_optim = torch.optim.Adam([self.w], lr=0.001)
                        self.feat_refs = None  # referências de características
                        self.points0_pt = None # referências de pontos                        
                        
                    # Zera os gradientes dos pesos do otimizador antes de calcular os novos gradientes. 
                    # Isso é necessário porque, por padrão, os gradientes no PyTorch se acumulam.                
                    self.w_optim.zero_grad()
                    
                    # Calcula os gradientes da perda em relação a todas as variáveis que requerem gradiente (definidas por requires_grad=True).
                    loss.backward()

                    # Aplica uma atualização aos parâmetros com base nos gradientes calculados. 
                    # Esta é a etapa onde o código latente ws é efetivamente ajustado na direção que minimiza a perda.
                    # Aplica uma etapa de otimização (atualização dos pesos) usando os gradientes calculados 
                    # pelo loss.backward(). 
                    self.w_optim.step()               
            else:
            
                if not self.exibiu_log:
                    self.exibiu_log = True
                    # Medir o tempo de execução
                    end_time = time.time() # Captura o tempo final
                    elapsed_time = end_time - self.start_time  # Calcula a duração em segundos
                    
                    # Medir o consumo de memória
                    self.gpu_memory_allocated_after = torch.cuda.memory_allocated()
                    self.gpu_memory_reserved_after = torch.cuda.memory_reserved()
                    
                    gpu_memory_allocated_used = self.gpu_memory_allocated_after - self.gpu_memory_allocated_before
                    gpu_memory_reserved_used = self.gpu_memory_reserved_after - self.gpu_memory_reserved_before


                    # Obtendo informações de memória
                    memory_info = psutil.virtual_memory()

                    # Memória total (em bytes)
                    total_ram_memory = memory_info.total / (1024 ** 2)  # Convertendo para MB

                    # Memória utilizada (em bytes)
                    used_ram_memory = memory_info.used / (1024 ** 2)  # Convertendo para MB

                    # Memória disponível (em bytes)
                    available_ram_memory = memory_info.available / (1024 ** 2)  # Convertendo para MB

                    # Percentual de memória utilizada
                    percent_used = memory_info.percent

                    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    current_memory_allocated = torch.cuda.memory_allocated(0)
                    current_memory_reserved = torch.cuda.memory_reserved(0)

                    # Convertendo bytes para megabytes
                    total_gpu_memory_mb = total_gpu_memory / (1024 ** 2)
                    current_memory_allocated_mb = current_memory_allocated / (1024 ** 2)
                    current_memory_reserved_mb = current_memory_reserved / (1024 ** 2)

                    # Imprimindo informações

                    print(50 * "*")
                    
                    dist = ""                    
                    for d in self.dist_ini_pi_ti:
                        dist += str(d) + " "
                    print(f"Distância inicial entre pi e ti: {dist}") 

                    print(f"Total de Iterações Tradicionais utilizadas: {self.num_tot_iteracoes_trad}") 
                    print(f"Total de Iterações Propostas utilizadas: {self.num_tot_iteracoes_prop }") 
                    print(f"Tempo decorrido: {elapsed_time} segundos")  # Exibe o tempo decorrido
                    
                    print(f"\nMemória da GPU alocada usada: {gpu_memory_allocated_used / 1024**2} MB")
                    print(f"Memória da GPU reservada usada: {gpu_memory_reserved_used / 1024**2} MB")
                    print(f"Memória total da GPU: {total_gpu_memory_mb:.2f} MB")
                    print(f"Memória da GPU atualmente alocada: {current_memory_allocated_mb:.2f} MB")
                    print(f"Memória da GPU atualmente reservada: {current_memory_reserved_mb:.2f} MB")
                    print(f"Memória RAM Total: {total_ram_memory:.2f} MB")
                    print(f"Memória RAM Usada: {used_ram_memory:.2f} MB")
                    print(f"Memória RAM Disponível: {available_ram_memory:.2f} MB")
                    print(f"Percentual de Memória RAM Usada: {percent_used}%")

                    print(50 * "*")

        #print(f"\nself.num_iteracoes: {self.num_iteracoes}") 
        
        self.num_iteracoes += 1

        #----------------------------------------------------------------------------
        # Dimensione e converta para uint8 (Seleção e Normalização da Imagem)
        #----------------------------------------------------------------------------

        # Pega a primeira imagem do lote (no nosso caso há apenas 1 lote)
        img = img[0]

        if img_normalize: # False por padrão
            # Normaliza a imagem
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)

        # Ajusta a escala da imagem
        img = img * (10 ** (img_scale_db / 20)) # img_scale_db = 0 por padrão

        # Ajusta a escala e o offset dos valores dos pixels para que estejam no intervalo [0, 255]
            # Clampa os valores: para garantir que todos os valores de pixels estejam dentro do intervalo [0, 255].
            # Converte os valores para uint8: o formato padrão para imagens com valores de pixel entre 0 e 255
            # Reordena as dimensões: usando permute(1, 2, 0)
                # para reorganizar as dimensões da matriz de imagem de um formato específico do PyTorch para um formato mais padrão (altura, largura, canais)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        
        if to_pil: # False por padrão 
            # Converte a imagem em um formato utilizável pela biblioteca Python Imaging Library (PIL)
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img) # Converte o array NumPy para um objeto de imagem PIL
        
        # Armazena a imagem processada no objeto res, que parece ser um container para os resultados do processo
        res.image = img
        
        # Armazena o vetor latente ws (depois de desanexá-lo do gráfico de computação, movendo-o para a CPU e convertendo-o para NumPy) no objeto res
        res.w = ws.detach().cpu().numpy()

#----------------------------------------------------------------------------
