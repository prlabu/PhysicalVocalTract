# python view_rt_lpc.py -r 16000 -i 50
# python view_rt_lpc.py -r 16000 -b 500 -i 500


"""Plot real-time LPC estimation.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

import librosa
import scipy.signal as scisig


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=100,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--block_duration', type=float, metavar='DURATION', default=100,
    help='block size (default %(default)s milliseconds)')   
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
 
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global plotdata
    # data = np.squeeze(data)
    indata = np.squeeze(indata)
    print(indata.shape)
    a = librosa.lpc(indata, 20)
    b = 1        
    f, h = scisig.freqz(b, a, worN=256, fs=args.samplerate)    
    plotdata = 20 * np.log10(abs(h))


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    # while True:

    # try:
        
    #     else:
    #         return lines
    #     # if q.qsize() > (args.samplerate/1000 * 30):
    #     #     data = q.get_nowait()
    #     # else: 
    #     #     break
    # except queue.Empty:
    #     break
    # shift = len(data)
    # plotdata = np.roll(plotdata, -shift, axis=0)
    # plotdata[-shift:, :] = data
    # if q.full(): 
    #     data = q.get_nowait()
    # else: 
    #     return lines

    lines[0].set_ydata(plotdata)

    # lines[0].set_ydata(plotdata)
    # lines[0].set_ydata(np.random.rand(lines[0].get_ydata().shape[0]))

    # for column, line in enumerate(lines):
    #     line.set_ydata(plotdata[:, column])

    return lines


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    a = [ 1.        , -2.2256038 ,  2.0256052 , -1.5742987 ,  0.5937929 ,
       -0.03303174,  0.6908514 , -0.6110801 ,  0.41314426, -0.19244389,
        0.0636103 , -0.2615614 ,  0.11687059,  0.04017332, -0.12633257,
       -0.00290638,  0.06697404,  0.0492343 , -0.18725695,  0.28708807,
       -0.13088977] 
    b = 1
    f, h = scisig.freqz(b, a, worN=256, fs=args.samplerate)

    print(f'Sampling rate: {args.samplerate}')

    #  = int(args.window * args.samplerate / (1000 * args.downsample))
    # plotdata = np.zeros((length, len(args.channels)))
    plotdata = 20 * np.log10(abs(h))

    fig, ax = plt.subplots()
    lines = ax.plot(f, np.zeros(plotdata.shape), 'k')
    # if len(args.channels) > 1:
    #     ax.legend(['channel {}'.format(c) for c in args.channels],
    #               loc='lower left', ncol=len(args.channels))
    ax.axis((f[0], f[-1], -20, 70))
    # ax.set_yticks([0])
    ax.yaxis.grid(True)
    # ax.tick_params(bottom=False, top=False, labelbottom=False,
    #                right=False, left=False, labelleft=False)
    ax.tick_params(bottom=True, top=False, labelbottom=True,
                   right=False, left=True, labelleft=True)               
    # fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=1,
        samplerate=args.samplerate, callback=audio_callback, 
        blocksize=int(args.samplerate * args.block_duration / 1000))
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    # ani = FuncAnimation(fig, update_plot, blit=True)
    with stream:
        plt.show()

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))