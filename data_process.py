import pywt
import scipy.fftpack as sf
import os
import numpy as np

from scipy import signal
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Lock

lock = Lock()


def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(sf.fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result


def dataWave(data, name):
    try:
        lock.acquire()
        if not os.path.exists('./images/waveform/'):
            os.makedirs('./images/waveform/')
        if 'WT' in name:
            data = data[0]
        plt.figure(f'{name}_waveform')
        plt.plot(data)
        plt.xlabel('采样点', fontsize=12)
        plt.ylabel('幅度', fontsize=12)
        plt.savefig(f"./static/waveform/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def spectrogram(data, name):
    try:
        lock.acquire()
        # 创建保存频谱图的文件夹，画出信号的频谱图并保存为png文件
        if not os.path.exists('./images/spectrogram/'):
            os.makedirs('./images/spectrogram/')
        if 'WT' in name:
            data = data[0]
        x, Spectrogram = FFT(1, data)
        plt.figure(f'{name}_spectrogram')
        plt.plot(Spectrogram)
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('幅度', fontsize=12)
        plt.savefig(f"./static/spectrogram/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def stft(data, name):
    try:
        lock.acquire()
        # 创建保存时频图的文件夹，画出信号的时频图并保存为png文件
        if not os.path.exists('./images/stft/'):
            os.makedirs('./images/stft/')
        if 'WT' in name:
            data = data[0]
        fre, t, zxx = signal.stft(data, nperseg=16)  # 短时傅里叶变换
        plt.figure(f'{name}_stft')
        plt.pcolormesh(t, fre, np.abs(zxx), shading='auto')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('频率 (Hz)', fontsize=12)
        plt.savefig(f"./images/stft/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def eliminateOutlier(data):
    """1.异常值剔除"""
    # 异常值剔除
    from sklearn.neighbors import LocalOutlierFactor

    predictions = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(data.reshape(-1, 1)).predict(data.reshape(-1, 1))
    data2 = data[predictions == 1]
    return data2


def smooth(data):
    """2.数据平滑"""
    # 数据平滑
    # 使用移动平均滤波器平滑数据
    wnd_size = 10
    wnd = np.ones(wnd_size) / wnd_size
    smooth_data = np.convolve(data, wnd, mode="same")
    return smooth_data


def normalize(data):
    """3.数据归一化"""
    # 归一化
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def WT(data):
    """4.小波变换"""
    try:
        # 小波变换
        WT_data, _ = pywt.cwt(
            data, scales=np.arange(1, 64), wavelet="cmor3-3", sampling_period=1.0
        )
        return WT_data
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)


def signalFilter(data):
    """5.信号滤波"""

    # 带通滤波
    Fs = 400000000  # Hz
    # 其中第一个4表示阶数  []里面的分别表示滤波的下限和上限
    b, a = signal.butter(4, [4 / (Fs / 2), 110000000 / (Fs / 2)], "bandpass")

    # 对上述数据进行带通滤波
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


if __name__ == '__main__':
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


