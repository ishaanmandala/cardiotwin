---
title: CardioTwin
emoji: 🫀
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Heart-scan estimates from a 10-second ECG (research)
---

# CardioTwin

Estimates what an echocardiogram (heart ultrasound) would show from a 10-second ECG, and
draws the result as a digital twin of the heart.

**Research demo, not a medical device.** It has not been clinically validated and must not be
used to diagnose or treat anyone.

Source code, methods and results: https://github.com/ishaanmandala/cardiotwin

Trained on the MUSIC dataset (PhysioNet, ODbL; Martin-Yebra A, Martinez JP, Laguna P,
doi:10.13026/z3m7-rf58), with an encoder pretrained on PTB-XL (CC BY 4.0; Wagner P et al.,
doi:10.13026/kfzx-aw45).
