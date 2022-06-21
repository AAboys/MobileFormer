# Mobile-Former: Pytorch Implementation

This is a PyTorch implementation of the paper [Mobile-Former: Bridging MobileNet and Transformer](https://arxiv.org/abs/2108.05895):
```
@Article{MobileFormer2021,
  author  = {Chen, Yinpeng and Dai, Xiyang and Chen, Dongdong and Liu, Mengchen and Dong, Xiaoyi and Yuan, Lu and Liu, Zicheng},
  journal = {arXiv:2108.05895},
  title   = {Mobile-Former: Bridging MobileNet and Transformer},
  year    = {2021},
}
```

* This repo is based on [`timm==0.3.4`](https://github.com/rwightman/pytorch-image-models).

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model</th>
<th valign="bottom">Input</th>
<th valign="bottom">Param</th>
<th valign="bottom">FLOPs</th>
<th valign="bottom">Top-1</th>
<th valign="bottom">Pretrained</th>
<!-- TABLE BODY -->
<tr><td align="left">mobile-former-508m</td>
<td align="center">224</td>
<td align="center">14.0M</td>
<td align="center">508M</td>
<td align="center">79.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1bqLIcpbCaxK-Eb4wcxk0iJpFJ1nXfGuF/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">mobile-former-294m</td>
<td align="center">224</td>
<td align="center">11.4M</td>
<td align="center">294M</td>
<td align="center">77.9</td>
<td align="center"><a href="https://drive.google.com/file/d/1JBSM7NJ60fN9TgT5sMnbhmbZwrzcBd0r/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">mobile-former-214m</td>
<td align="center">224</td>
<td align="center">9.4M</td>
<td align="center">214M</td>
<td align="center">76.7</td>
<td align="center"><a href="https://drive.google.com/file/d/13MOjdIMMBJgqyq_ld-grRhBBHOFU5Ggd/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">mobile-former-151m</td>
<td align="center">224</td>
<td align="center">7.6M</td>
<td align="center">151M</td>
<td align="center">75.2</td>
<td align="center"><a href="https://drive.google.com/file/d/1w4QO-zhJ8QI1zGu7iOZ7lXxYJ5G1VXdF/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">mobile-former-96m</td>
<td align="center">224</td>
<td align="center">4.6M</td>
<td align="center">96M</td>
<td align="center">72.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1kMbRceswujtliCHKr-nxMFsrkvBrEfrW/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">mobile-former-52m</td>
<td align="center">224</td>
<td align="center">3.5M</td>
<td align="center">52M</td>
<td align="center">68.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1ekq_FPl57gjIlYX16Ll0nBuEyB0pjgGt/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">mobile-former-26m</td>
<td align="center">224</td>
<td align="center">3.2M</td>
<td align="center">26M</td>
<td align="center">64.0</td>
<td align="center"><a href="https://drive.google.com/file/d/15uWYzx2VWWUjacZQHB56vOHzKxQDlp5O/view?usp=sharing">download</a></td>
</tr>
</tbody></table>
