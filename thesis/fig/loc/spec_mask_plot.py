import os
import sys
import soundfile as sf
import numpy as np
from ipdb import set_trace

sys.path.append('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/ssivasankaran/experiments/code/sunit-code/utils')
import stft
import plt_spec
WINDOW_LEN = 1600
stft_if = stft.STFT(WINDOW_LEN)
eps = np.finfo('float').eps

base_path = '/home/ssivasan/Dropbox/sunit_inria/phd/thesis/SunitThesis/thesis/fig/ss/wav'
wav_id = "447o030d_2.1006_050a0506_-2.1006.wav"

s1 = os.path.join(base_path , 's1', wav_id)
s2 = os.path.join(base_path , 's2', wav_id)

s1_data = sf.read(s1, always_2d=True)[0][:,0]
s2_data = sf.read(s2, always_2d=True)[0][:,0]
s1_data = np.asfortranarray(s1_data)
s2_data = np.asfortranarray(s2_data)
mix = s1_data + s2_data
mix = mix/np.sum(s1_data**2)
mix_stft = np.abs(stft_if.compute_stft(mix))
s1_stft = np.abs(stft_if.compute_stft(s1_data))
s2_stft = np.abs(stft_if.compute_stft(s2_data))
mask = s1_stft/(s1_stft + s2_stft + 1e-7)
set_trace()
mix_stft = 20 * np.log10(mix_stft+1e-2)
mix_stft = mix_stft[..., :100]
mask = mask[..., :100]

plt_spec.FullSpec(mix_stft, '/tmp/mix_nc.pdf', cbarTitle='Magnitude (dB)', clim=(-30, 0))
plt_spec.FullSpec(mask, '/tmp/mask_nc.pdf', clim=(0,1))
os.system('pdfcrop /tmp/mix_nc.pdf /tmp/mix.pdf')
os.system('pdfcrop /tmp/mask_nc.pdf /tmp/mask.pdf')
