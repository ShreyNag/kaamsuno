# VAD comparison (held-out clips only)

Data: `data\vad_dev` (12 clips). Noises: drilling, generator, hum50, pink, traffic, white (from `data\noise_16k` + generated); SNRs 10dB, 5dB. Frames of 10 ms; metrics pooled over frames (micro). Tuning objective: F2 on the other clips.

## Overall

| VAD | view | precision | recall | F1 | F2 | false alarm (noise-only) | knob chosen (per fold) |
|---|---|---:|---:|---:|---:|---:|---|
| custom | raw | 0.962 | 0.673 | 0.792 | 0.716 | 0.112 | (0.2, 0.45, 0.35, 1, 0.5) |
| custom | deployed | 0.976 | 0.726 | 0.832 | 0.765 | 0.076 | (0.2, 0.45, 0.35, 1, 0.5) |
| silero | raw | 0.996 | 0.856 | 0.921 | 0.881 | 0.013 | 0.3 |
| silero | deployed | 0.976 | 0.839 | 0.902 | 0.863 | 0.085 | 0.3 |
| webrtc | raw | 0.938 | 0.867 | 0.901 | 0.880 | 0.236 | 0 |
| webrtc | deployed | 0.945 | 0.862 | 0.902 | 0.877 | 0.209 | 0 |
| composite | raw | 1.000 | 0.522 | 0.686 | 0.577 | 0.001 | (0.3, 0.34) |
| composite | deployed | 0.994 | 0.546 | 0.705 | 0.600 | 0.014 | (0.3, 0.34) |

## Recall by condition (raw view)

| condition | custom | silero | webrtc | composite |
|---|---:|---:|---:|---:|
| clean | 0.818 | 0.867 | 0.947 | 0.732 |
| drilling@10dB | 0.670 | 0.862 | 0.882 | 0.527 |
| drilling@5dB | 0.578 | 0.866 | 0.877 | 0.420 |
| generator@10dB | 0.730 | 0.848 | 0.893 | 0.537 |
| generator@5dB | 0.655 | 0.845 | 0.913 | 0.517 |
| hum50@10dB | 0.656 | 0.842 | 0.920 | 0.518 |
| hum50@5dB | 0.562 | 0.834 | 0.919 | 0.466 |
| pink@10dB | 0.793 | 0.864 | 0.829 | 0.587 |
| pink@5dB | 0.762 | 0.861 | 0.784 | 0.551 |
| traffic@10dB | 0.697 | 0.856 | 0.827 | 0.534 |
| traffic@5dB | 0.631 | 0.857 | 0.796 | 0.482 |
| white@10dB | 0.646 | 0.864 | 0.859 | 0.507 |
| white@5dB | 0.550 | 0.859 | 0.820 | 0.409 |

## False-alarm rate on noise-only frames by condition (raw view)

| condition | custom | silero | webrtc | composite |
|---|---:|---:|---:|---:|
| clean | 0.012 | 0.028 | 0.181 | 0.008 |
| drilling@10dB | 0.038 | 0.022 | 0.154 | 0.000 |
| drilling@5dB | 0.054 | 0.030 | 0.208 | 0.000 |
| generator@10dB | 0.171 | 0.002 | 0.263 | 0.000 |
| generator@5dB | 0.195 | 0.015 | 0.502 | 0.000 |
| hum50@10dB | 0.000 | 0.000 | 0.247 | 0.000 |
| hum50@5dB | 0.000 | 0.001 | 0.313 | 0.000 |
| pink@10dB | 0.399 | 0.008 | 0.092 | 0.002 |
| pink@5dB | 0.397 | 0.016 | 0.196 | 0.000 |
| traffic@10dB | 0.083 | 0.010 | 0.200 | 0.000 |
| traffic@5dB | 0.102 | 0.021 | 0.330 | 0.002 |
| white@10dB | 0.000 | 0.007 | 0.182 | 0.000 |
| white@5dB | 0.000 | 0.015 | 0.202 | 0.000 |

## F1 by condition (raw view)

| condition | custom | silero | webrtc | composite |
|---|---:|---:|---:|---:|
| clean | 0.898 | 0.925 | 0.951 | 0.844 |
| drilling@10dB | 0.798 | 0.923 | 0.919 | 0.690 |
| drilling@5dB | 0.727 | 0.925 | 0.910 | 0.591 |
| generator@10dB | 0.824 | 0.918 | 0.913 | 0.698 |
| generator@5dB | 0.770 | 0.914 | 0.898 | 0.682 |
| hum50@10dB | 0.792 | 0.914 | 0.930 | 0.682 |
| hum50@5dB | 0.720 | 0.910 | 0.922 | 0.636 |
| pink@10dB | 0.840 | 0.926 | 0.896 | 0.740 |
| pink@5dB | 0.820 | 0.924 | 0.857 | 0.710 |
| traffic@10dB | 0.812 | 0.921 | 0.882 | 0.696 |
| traffic@5dB | 0.762 | 0.921 | 0.849 | 0.651 |
| white@10dB | 0.785 | 0.926 | 0.903 | 0.673 |
| white@5dB | 0.710 | 0.923 | 0.878 | 0.581 |

Raw = per-chunk decision; deployed = after Pipecat's start/stop counters (0.2/0.2 s), what the turn logic sees. Labels from `eval/auto_label.py` are padded generously, so precision/false alarm near segment edges is pessimistic.