# Kalibrasyon doğrulaması: sanal havuz (REMUS 100)

Bu dosya `validation/calibration_remus.py` ile üretilir, elle düzenleme.

**Soru:** Araç hareket kayıtlarından, CAD'in yanlış tahmin ettiği sönümü geri bulabilir miyiz?

- **Gerçek araç (sanal):** REMUS 100, Prestero'nun deniz testlerine göre ayarladığı
  katsayılarla (Ek C). robomodel'in CAD değerlerine göre çarpan olarak yazıldı.
- **Ölçüm:** bu araçla iki test (zikzak dümen, dalış dümeni), 40 s, 10 Hz; derinlik,
  açılar, hızlar ve açısal hızlara tipik sensör gürültüsü eklendi.
- **Kalibrasyon:** sadece CAD'den çıkan model, bu iki kayda uyduruldu.
- **Kontrol:** kalibrasyonda kullanılmayan üçüncü bir kayıt ve deniz testindeki dönüş hızı.

## Bulunan çarpanlar

| Terim | Gerçek | Kalibrasyonla bulunan | Fark | Belirsizlik (1σ) |
|---|---|---|---|---|
| Sönüm, surge | ×1.02 | ×1.01 | -0% | ±0% |
| Sönüm, sway | ×10.2 | ×9.78 | -4% | ±6% |
| Sönüm, heave | ×10.2 | ×10.1 | -1% | ±1% |
| Sönüm, pitch | ×15.7 | ×15.7 | -0% | ±0% |
| Sönüm, yaw | ×12.6 | ×12.6 | +1% | ±1% |
| Kanat kaldırması | ×0.505 | ×0.502 | -1% | ±0% |
| Gövde kaldırması | ×1.27 | ×1.29 | +2% | ±1% |

Yuvarlanma sönümü bu testlerde uyarılmadığı için CAD değerinde kaldı (tabloda yok).

## Kalibrasyonda kullanılmayan kayıtta uyum (%, 100 = birebir)

| Kanal | Sadece CAD | Kalibrasyon sonrası | Tavan (gerçek aracın kendisi) |
|---|---|---|---|
| `z` | -149 | 99 | 99 |
| `phi` | -6 | 1 | 1 |
| `theta` | -162 | 80 | 80 |
| `psi` | -226 | 93 | 93 |
| `u` | -180 | 88 | 88 |
| `v` | -90 | 6 | 6 |
| `w` | -852 | 3 | 3 |
| `p` | -105 | 12 | 12 |
| `q` | -107 | 53 | 53 |
| `r` | -145 | 63 | 63 |
| **ortalama (psi, r, z, theta)** | **-171** | **84** | **84** |

Tavan: gerçek aracın kendi modeli bile gürültülü ölçüme bundan iyi uyamaz. Hareket
küçük olan kanallarda (`phi`, `p`, `v`, `w`) sinyal gürültüden küçük olduğu için yüzde
düşük çıkar; bu kanallar ortalamaya alınmadı.

## Dönüş testi: 4° dümen, 1,54 m/s

| Model | Dönüş hızı (°/s) | Sanal gerçek araca göre |
|---|---|---|
| Sanal gerçek araç | 2.6 | – |
| Sadece CAD | 15.9 | +517% |
| Kalibre edilmiş | 2.6 | -0% |

**Bu test neyi gösterir, neyi göstermez:** kalibrasyon algoritmasının, bilinen bir
gerçeği gürültülü hareket verisinden geri bulduğunu gösterir. Gerçek REMUS'u göstermez:
Prestero'nun ayarlı katsayıları bizim denklemlerimizde 2.6°/s veriyor,
denizde ölçülen ~10°/s değil (onun modelinde bizde olmayan çapraz terimler var ve
kanat tablosu kendi formülüyle tutarsız). Gerçek araçta kalibrasyon havuz verisiyle
yapılmalı; o zaman model yapısının kendi eksikleri de sonuca girer.

```
Calibration (scale x CAD value, ± relative 1 sigma)
  damping surge  x1.01 ± 0%
  damping sway   x9.78 ± 6%
  damping heave  x10.1 ± 1%
  damping roll   x1.16 ± 1%
  damping pitch  x15.7 ± 0%
  damping yaw    x12.6 ± 1%
  fin lift       x0.502 ± 0%
  body lift      x1.29 ± 1%

Calibration logs (fit %, 100 = perfect)
  zigzag: z -165->99, phi -8->7, theta -178->78, psi -977->90, u -177->88, v -102->19, w -903->3, p -149->41, q -158->34, r -93->77
  dalis: z -125->99, phi -0->-0, theta -138->84, psi -0->-0, u -119->90, v -0->-0, w -861->3, p -0->-0, q -128->55, r -0->-0

Validation logs (fit %, 100 = perfect)
  dogrulama: z -149->99, phi -6->1, theta -162->80, psi -226->93, u -180->88, v -90->6, w -852->3, p -105->12, q -107->53, r -145->63
```
