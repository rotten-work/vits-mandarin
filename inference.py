import os
import time

import numpy as np
import torch
from scipy.io.wavfile import write
import soundfile as sf

import commons
import utils
from models import SynthesizerTrn
from text import create_symbols_manager, text_to_sequence

import argparse

class AudioGenerator():
    def __init__(self, hparams, device):
        self.hparams = hparams
        self._device = device

        symbols_manager = create_symbols_manager(hparams.data.language)
        self.symbol_to_id = symbols_manager._symbol_to_id

        self.net_g = create_network(hparams, symbols_manager.symbols, device)

    def load(self, path):
        load_checkpoint(self.net_g, path)

    def inference(self, text):
        return do_inference(self.net_g, self.hparams, self.symbol_to_id, text, self._device)

def get_text(text, hparams, symbol_to_id):
    text_norm = text_to_sequence(text, hparams.data.text_cleaners, symbol_to_id)
    if hparams.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def create_network(hparams, symbols, device):
    net_g = SynthesizerTrn(
        len(symbols),
        hparams.data.filter_length // 2 + 1,
        hparams.train.segment_size // hparams.data.hop_length,
        **hparams.model).to(device)
    _ = net_g.eval()

    return net_g

def load_checkpoint(network, path):
    _ = utils.load_checkpoint(path, network, None)

# Assume the network has loaded weights and are ready to do inference
def do_inference(generator, hparams, symbol_to_id, text, device):
    stn_tst = get_text(text, hparams, symbol_to_id)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        noise_scale = 0.667
        noise_scale_w = 0.8
        audio = generator.infer(x_tst, x_tst_lengths, noise_scale, noise_scale_w, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    return audio

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
def save_to_wav_1(audio, sampling_rate, path):
    max = 32767
    audio_int16 = np.floor(((max + 1) * audio)).astype(np.int16)
    write(path, sampling_rate, audio_int16)

def save_to_wav(data, sampling_rate, path):
    sf.write(path, data, 22050, 'PCM_16')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str)
    parser.add_argument('-n', '--name', type=str, default="0")
    parser.add_argument('-g', '--gpu', action="store_true")

    # args, leftovers = parser.parse_known_args()
    args = parser.parse_args()

    # https://stackoverflow.com/questions/53331247/pytorch-0-4-0-there-are-three-ways-to-create-tensors-on-cuda-device-is-there-s/53332659#53332659
    # https://community.esri.com/t5/arcgis-image-analyst-questions/how-force-pytorch-to-use-cpu-instead-of-gpu/td-p/1046738
    # torch.cuda.is_available = lambda : False
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')

    if args.gpu:
        print("Use GPU")
        device = torch.device('cuda')
    else:
        print("Use CPU")
        device = torch.device('cpu')

    config_path = "./configs/ljs_windows.json"
    # config_path = "./configs/bb_v100.json"
    # config_path = "./configs/kkr_tiny_laptop.json"
    # config_path = "./configs/inference_ce.json"
    hps = utils.get_hparams_from_file(config_path)

    audio_generator = AudioGenerator(hps, device)

    # checkpoint_path = "./models/G_lex_orig_laptop_5000.pth"
    # checkpoint_path = "./models/G_bb_v100_820000.pth"
    # checkpoint_path = "./models/G_kkr_tiny_laptop_7000.pth"
    checkpoint_path = "./models/G_ljs_windows_783000.pth"
    audio_generator.load(checkpoint_path)

    if args.filepath is not None:
        print("Batch generation:")

        output_dir = os.path.join('./output/', args.name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.filepath, encoding='utf-8') as f:
            texts = [line.strip() for line in f]

        print(texts)

        start = time.perf_counter()
        index = 1
        for text in texts:
            audio = audio_generator.inference(text)
            filename = f"{args.name}_{index:04}.wav"
            output_path = os.path.join(output_dir, filename)
            save_to_wav(audio, hps.data.sampling_rate, output_path)
            index += 1
        print(f"The inference takes {time.perf_counter() - start} seconds")
    else:
        # text = "我是御坂妹妹。"
        # text = "我是你爹，喵喵抽风！"
        # text = "喵喵抽风，是乱杀之星！"
        # text = "炸鸡，是喵喵抽风的大儿子。"
        # text = "我了解，你在人鱼港当了团长，生活的很好，有剧团和艾丽卡保护你，你不需要我这种舰娘。"
        # text = "但是，你现在来找我说，列克星敦，我喜欢你？我到底做错了什么让你这么不尊重我？"
        # text = "卡尔普陪外孙玩滑梯。"
        # text = "他的到来是一件好事，我很欢迎他，不管是代表个人，还是代表俱乐部。"
        # text = "研究完成，dou， 您可以制定新的科研方向了，司令官。"
        # text = "哎！司令官，就像个孩子一样调皮呢！嗯哼！不过，还挺可爱的。"
        # text = "12345！"
        
        # text = "高い山のいただきに住んで、小鳥を取って食べたり、"

        # text = "yi2 jian4  san1 lian2！" 

        # text = "It has used other Treasury law enforcement agents on special experiments in building and route surveys in places to which the President frequently travels."

        text = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
        
        start = time.perf_counter()
        audio = audio_generator.inference(text)
        print(f"The inference takes {time.perf_counter() - start} seconds")

        print(audio.dtype)

        output_dir = './output/'
        # python program to check if a path exists
        # if it doesn’t exist we create one
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = 'output.wav'
        file_path = os.path.join(output_dir, filename)

        save_to_wav(audio, hps.data.sampling_rate, file_path)


