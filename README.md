# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)

<br>

## 补充说明
### 项目特点
- 支持Windows和Linux，两个平台上都可以进行训练和推断
- 兼容最新版本的各个依赖库
- Windows平台所需特殊环境配置和操作说明
- 支持中文和英文
- 本项目添加了一个简易的面向对象风格的[推断脚本](inference.py)。
- [这里](https://colab.research.google.com/drive/1uFUnZDbHMqKWBUQDZKih56Vkj2ixTN9B)是一个简单的Colab notebook，展示了如何使用该项目进行训练和推断的步骤。
- [这里](https://colab.research.google.com/drive/1VWBOp3PDGNO77_xOm20yRtc4CSmsbqtb)是一个简单的Colab notebook，展示了如何使用预训练权重进行迁移训练（精调）
- 预处理好的几套音频数据集以方便大家学习实验


### Windows平台环境配置
#### 安装PyTorch的GPU版本
在Windows平台，<code>pip install -r requirements.txt</code> 安装的是CPU版本的PyTorch。所以需要去[PyTorch官网](https://pytorch.org)挑选并运行合适的GPU版本PyTorch安装命令。下面命令仅供参考：
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### eSpeak的配置
- 在Windows平台上用英文做训练或推断的话，需要安装[eSpeak Ng](https://github.com/espeak-ng/espeak-ng)库。[这里](https://github.com/espeak-ng/espeak-ng/releases)是下载页面，推荐使用.msi安装。
- 安装eSpeak Ng后，请添加环境变量PHONEMIZER_ESPEAK_LIBRARY，并将变量值设置为{INSTALLDIR}\libespeak-ng.dll。如图所示：<br>
<img src="resources/PHONEMIZER_ESPEAK_LIBRARY.png">

#### 构建Monotonoic Alignment Search扩展模块
请先下载安装Visual Studio。到[这里](https://visualstudio.microsoft.com/#vs-section)下载。

### 数据集
<table style="width:100%">
  <tr>
    <td>标贝中文标准女声音库（处理后）16-bit PCM WAV，22050 Hz</td>
    <td>
      链接：https://pan.baidu.com/s/1oihti9-aoJ447l54kdjChQ <br>
      提取码：vits 
    </td>
  </tr>
  <tr>
    <td>LJSpeech数据集16-bit PCM WAV，22050 Hz</td>
    <td>
      链接：https://pan.baidu.com/s/1q2A38znFmxn3zCn587ZKkw <br>
      提取码：vits
    </td>
  </tr>
  <tr>
    <td>标贝中文标准女声音库官网</td>
    <td>https://www.data-baker.com/data/index/TNtts/</td>
  </tr>
  <tr>
    <td>LJSpeech数据集官网</td>
    <td>https://keithito.com/LJ-Speech-Dataset/</td>
  </tr>
</table>
<br>

### 预训练权重
<table style = "width:100%">
  <tr>
    <td>标贝中文标准女声音库预训练权重</td>
    <td>
      链接：https://pan.baidu.com/s/1pN-wL_5wB9gYMAr2Mh7Jvg <br>
      提取码：vits
    </td>
  </tr>
</table>
注：各预训练权重文件包括生成网络权重（G开头），鉴别器网络权重（D开头），还有训练时使用的cleaners与symbols（方便与其他VITS仓库的代码或工具兼容）<br><br>

## 效果展示
### [Gallery](gallery/Gallery.md) <br><br>

## 参考与鸣谢
### 大佬们的VITS语音合成GitHub仓库
*   https://github.com/jaywalnut310/vits
*   https://github.com/CjangCjengh/vits
*   https://github.com/AlexandaJerry/vits-mandarin-biaobei
*   https://github.com/JOETtheIV/VITS-Paimon
*   https://github.com/w4123/vits
*   https://github.com/xiaoyou-bilibili/tts_vits
*   https://github.com/wind4000/vits.git
### 参考B站链接
*   【CV失业计划】基于VITS神经网络模型的近乎完美派蒙中文语音合成：\
  https://www.bilibili.com/video/BV1rB4y157fd
*   【原神】派蒙Vtuber出道计划——基于AI深度学习VITS和VSeeFace的派蒙语音合成/套皮：\
https://www.bilibili.com/video/BV16G4y1B7Ey
*   【深度学习】基于vits的语音合成：\
https://www.bilibili.com/video/BV1Fe4y1r737
*   零基础炼丹 - vits版补充：\
https://www.bilibili.com/read/cv18357171


## 恰饭
生活不易，喵喵叹气。。。如果您喜欢该项目，请对该项目star一下表示支持吧~ <br> <img src="resources/恰饭512.png">
