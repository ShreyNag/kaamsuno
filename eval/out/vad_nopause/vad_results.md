# VAD comparison (held-out clips only)

Data: `data\vad_dev_nopause` (6 clips). Noises: crowd, drilling, generator, hammering, hum50, pink, traffic, white (from `data\noise_16k` + generated); SNRs 10dB, 5dB. Frames of 10 ms; metrics pooled over frames (micro). Tuning objective: F2 on the other clips.

## Overall

| VAD | view | precision | recall | F1 | F2 | false alarm (noise-only) | knob chosen (per fold) |
|---|---|---:|---:|---:|---:|---:|---|
| custom | raw | 0.944 | 0.722 | 0.818 | 0.758 | 0.200 | (0.2, 0.45, 0.35, 1, 0.5) |
| custom | deployed | 0.975 | 0.798 | 0.878 | 0.828 | 0.094 | (0.2, 0.45, 0.35, 1, 0.5) |
| silero | raw | 0.993 | 0.896 | 0.942 | 0.914 | 0.027 | 0.3 |
| silero | deployed | 0.984 | 0.885 | 0.932 | 0.904 | 0.065 | 0.3 |
| webrtc | raw | 0.942 | 0.894 | 0.917 | 0.903 | 0.255 | 0 |
| webrtc | deployed | 0.961 | 0.893 | 0.925 | 0.905 | 0.170 | 0 |
| composite | raw | 0.997 | 0.562 | 0.719 | 0.616 | 0.008 | (0.3, 0.34) |
| composite | deployed | 0.996 | 0.603 | 0.752 | 0.655 | 0.010 | (0.3, 0.34) |

## Recall by condition (raw view)

| condition | custom | silero | webrtc | composite |
|---|---:|---:|---:|---:|
| clean | 0.825 | 0.878 | 0.955 | 0.735 |
| crowd@10dB | 0.831 | 0.939 | 0.908 | 0.637 |
| crowd@5dB | 0.806 | 0.955 | 0.953 | 0.578 |
| drilling@10dB | 0.692 | 0.894 | 0.888 | 0.570 |
| drilling@5dB | 0.611 | 0.895 | 0.887 | 0.444 |
| generator@10dB | 0.737 | 0.879 | 0.920 | 0.544 |
| generator@5dB | 0.693 | 0.876 | 0.940 | 0.524 |
| hammering@10dB | 0.804 | 0.899 | 0.899 | 0.597 |
| hammering@5dB | 0.808 | 0.887 | 0.931 | 0.618 |
| hum50@10dB | 0.672 | 0.880 | 0.928 | 0.545 |
| hum50@5dB | 0.588 | 0.874 | 0.931 | 0.486 |
| pink@10dB | 0.805 | 0.901 | 0.856 | 0.625 |
| pink@5dB | 0.775 | 0.896 | 0.809 | 0.575 |
| traffic@10dB | 0.730 | 0.900 | 0.857 | 0.580 |
| traffic@5dB | 0.660 | 0.898 | 0.838 | 0.519 |
| white@10dB | 0.667 | 0.895 | 0.855 | 0.545 |
| white@5dB | 0.573 | 0.896 | 0.840 | 0.437 |

## False-alarm rate on noise-only frames by condition (raw view)

| condition | custom | silero | webrtc | composite |
|---|---:|---:|---:|---:|
| clean | 0.014 | 0.019 | 0.170 | 0.009 |
| crowd@10dB | 0.362 | 0.125 | 0.314 | 0.032 |
| crowd@5dB | 0.457 | 0.193 | 0.542 | 0.035 |
| drilling@10dB | 0.041 | 0.007 | 0.109 | 0.000 |
| drilling@5dB | 0.064 | 0.000 | 0.145 | 0.000 |
| generator@10dB | 0.186 | 0.000 | 0.223 | 0.000 |
| generator@5dB | 0.296 | 0.000 | 0.557 | 0.000 |
| hammering@10dB | 0.293 | 0.000 | 0.246 | 0.000 |
| hammering@5dB | 0.690 | 0.084 | 0.559 | 0.048 |
| hum50@10dB | 0.000 | 0.000 | 0.174 | 0.000 |
| hum50@5dB | 0.000 | 0.000 | 0.342 | 0.000 |
| pink@10dB | 0.396 | 0.003 | 0.109 | 0.000 |
| pink@5dB | 0.391 | 0.007 | 0.126 | 0.000 |
| traffic@10dB | 0.101 | 0.007 | 0.135 | 0.000 |
| traffic@5dB | 0.110 | 0.016 | 0.338 | 0.004 |
| white@10dB | 0.000 | 0.003 | 0.109 | 0.000 |
| white@5dB | 0.000 | 0.003 | 0.141 | 0.000 |

## F1 by condition (raw view)

| condition | custom | silero | webrtc | composite |
|---|---:|---:|---:|---:|
| clean | 0.903 | 0.933 | 0.959 | 0.847 |
| crowd@10dB | 0.871 | 0.955 | 0.919 | 0.775 |
| crowd@5dB | 0.847 | 0.957 | 0.921 | 0.729 |
| drilling@10dB | 0.814 | 0.943 | 0.929 | 0.726 |
| drilling@5dB | 0.752 | 0.944 | 0.925 | 0.615 |
| generator@10dB | 0.829 | 0.935 | 0.935 | 0.704 |
| generator@5dB | 0.789 | 0.934 | 0.913 | 0.688 |
| hammering@10dB | 0.861 | 0.947 | 0.921 | 0.748 |
| hammering@5dB | 0.826 | 0.931 | 0.908 | 0.759 |
| hum50@10dB | 0.804 | 0.936 | 0.945 | 0.705 |
| hum50@5dB | 0.740 | 0.933 | 0.929 | 0.654 |
| pink@10dB | 0.852 | 0.948 | 0.911 | 0.770 |
| pink@5dB | 0.834 | 0.944 | 0.881 | 0.730 |
| traffic@10dB | 0.834 | 0.946 | 0.909 | 0.734 |
| traffic@5dB | 0.784 | 0.944 | 0.877 | 0.683 |
| white@10dB | 0.801 | 0.944 | 0.911 | 0.706 |
| white@5dB | 0.729 | 0.945 | 0.898 | 0.608 |

Raw = per-chunk decision; deployed = after Pipecat's start/stop counters (0.2/0.2 s), what the turn logic sees. Labels from `eval/auto_label.py` are padded generously, so precision/false alarm near segment edges is pessimistic.