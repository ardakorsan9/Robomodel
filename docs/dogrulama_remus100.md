# REMUS 100 doğrulaması: robomodel ve Prestero (2001)

Bu dosya `validation/remus100.py` ile üretilir, elle düzenleme.

Kaynak: T. Prestero, *Verification of a Six-Degree of Freedom Simulation Model for the
REMUS Autonomous Underwater Vehicle*, MIT/WHOI yüksek lisans tezi, 2001.
[MIT DSpace](https://dspace.mit.edu/handle/1721.1/65068)

- **Teori:** Prestero'nun el kitabı ve şerit teorisi tahmini (Tablo 4.2, 4.3).
  robomodel'in yaptığı tahminle aynı türden; yöntemimizi test eder.
- **Ayarlı / ölçülen:** deniz testlerine göre düzeltilmiş (Tablo 8.2, Ek C) ya da
  ölçülmüş değerler. Gerçeğe en yakın referans bu sütun.
- İşaretler Fossen/Prestero notasyonunda. Kütle, kütle merkezi ve eylemsizlik tezden
  ölçülmüş değer olarak girildi; karşılaştırmada sadece hesapladığımız terimler var.
- **Fark** büyüklük farkıdır: +%10, robomodel'in değeri %10 büyük demek. İşaret
  uyuşmazsa ayrıca yazılır.

| Grup | Katsayı | robomodel | Prestero teori | Fark | Prestero ayarlı / ölçülen | Fark | Kaynak | Not |
|---|---|---|---|---|---|---|---|---|
| Hidrostatik | `V (m^3)` | +0.0317 | +0.0315 | +0% | – | – | Tablo 2.7 | Prestero'nun gövde hacmi tahmini |
| Hidrostatik | `B (N)` | +320 | +317 | +1% | +306 | +4% | Tablo 2.3, 2.7 | ayarlı sütunu = ölçülen kaldırma; kanat, sonar ve su dolan burun hesapta yok |
| Hidrostatik | `x_cb burundan (m)` | +0.606 | – | – | +0.611 | -1% | Tablo 2.4 | ayarlı sütunu = ölçülen |
| Eklenmiş kütle | `X_udot` | -1.38 | -0.93 | +49% | -0.93 | +49% | Tablo 4.2, C.5 |  |
| Eklenmiş kütle | `Y_vdot` | -31 | -35.5 | -13% | -35.5 | -13% | Tablo 4.2, C.5 |  |
| Eklenmiş kütle | `Z_wdot` | -31 | -35.5 | -13% | -35.5 | -13% | Tablo 4.2, C.5 |  |
| Eklenmiş kütle | `K_pdot` | -0.0151 | -0.0141 | +7% | -0.0704 | -79% | Tablo 4.3, C.5 | Prestero denize göre 5 kat büyüttü |
| Eklenmiş kütle | `M_qdot` | -3.2 | -4.88 | -35% | -4.88 | -35% | Tablo 4.3, C.5 |  |
| Eklenmiş kütle | `N_rdot` | -3.2 | -4.88 | -35% | -4.88 | -35% | Tablo 4.3, C.5 |  |
| Eklenmiş kütle | `Y_rdot = N_vdot` | +0.588 | +1.93 | -70% | +1.93 | -70% | Tablo 4.2, C.5 |  |
| Eklenmiş kütle | `Z_qdot = M_wdot` | -0.588 | -1.93 | -70% | -1.93 | -70% | Tablo 4.2, C.5 |  |
| Sönümleme | `eksenel C_d` | +0.196 | +0.12 | +63% | +0.2 | -2% | Bölüm 4.2.1 | teori: el kitabı formülleri 0,11-0,13; ayarlı sütunu: denizde ölçülen 0,2 (havuzda 0,27, kanat ve sonar dahil) |
| Sönümleme | `X_u|u|` | -2.89 | -1.62 | +78% | -2.94 | -2% | Tablo 4.2, C.1 | tablo C_d ≈ 0,11'e karşılık geliyor; ayarlı sütunu: C_d = 0,2 |
| Sönümleme | `Y_v|v|` | -128 | -131 | -2% | -1.31e+03 | -90% | Tablo 4.2, C.2 | Prestero denize göre 10 kat büyüttü (Tablo 8.2) |
| Sönümleme | `Z_w|w|` | -128 | -131 | -2% | -1.31e+03 | -90% | Tablo 4.2, 8.2 | Tablo C.2'de -131 yazıyor, Tablo 8.2'ye göre 10 kat olmalı: tezde tutarsızlık |
| Sönümleme | `K_p|p|` | -0 | -0.0013 | -100% | -0.13 | -100% | Tablo 4.3, C.3 | bizde dairesel gövdede 0; kanatların çapraz akış sürüklemesi yok |
| Sönümleme | `M_q|q|` | -7.48 | -9.4 | -20% | -118 | -94% | Tablo 4.3, 8.2 | Prestero 12,5 kat büyüttü; Tablo C.2'de -188 yazıyor |
| Sönümleme | `N_r|r|` | -7.48 | -9.4 | -20% | -94 | -92% | Tablo 4.3, C.2 | Prestero 10 kat büyüttü |
| Kanat kaldırması | `Y_uvf` | -19.1 | -21.4 | -11% | -9.64 | +98% | denk. 4.44, Tablo C.10 | teori sütunu: tezin formülü (denk. 4.44-4.45); tablo bu formülden 2,2 kat küçük, tezde nedeni açıklanmıyor |
| Kanat kaldırması | `Y_urf` | +12.2 | +13.6 | -11% | +6.15 | +98% | denk. 4.44, Tablo C.10 |  |
| Kanat kaldırması | `N_uvf` | +12.2 | +13.6 | -11% | +6.15 | +98% | denk. 4.45, Tablo C.10 |  |
| Kanat kaldırması | `N_urf` | -7.77 | -8.7 | -11% | -3.93 | +98% | denk. 4.45, Tablo C.10 |  |
| Kanat kaldırması | `M_uqf` | -7.77 | -8.7 | -11% | -3.93 | +98% | denk. 4.45, Tablo C.10 |  |
| Kontrol | `Y_uudr` | +19.1 | +21.4 | -11% | +9.64 | +98% | denk. 4.44, Tablo C.10 |  |
| Kontrol | `N_uudr` | -12.2 | -13.6 | -11% | -6.15 | +98% | denk. 4.45, Tablo C.10 |  |
| Kontrol | `Z_uuds` | -19.1 | -21.4 | -11% | -9.64 | +98% | denk. 4.44, Tablo C.10 |  |
| Kontrol | `M_uuds` | -12.2 | -13.6 | -11% | -6.15 | +98% | denk. 4.45, Tablo C.10 |  |
| Gövde kaldırması | `Y_uvl` | – | -19 | – | -19 | – | Tablo 4.2 - C.10 | robomodel'de yok (Hoerner gövde kaldırması); Y_uv = -28,6 toplamından kanat payı çıkarıldı |

## Davranış: deniz testi

1,54 m/s hızda, iki dümen 4° açıyla, sabit dönüş hızı:

| Model | Dönüş hızı (°/s) | Deniz testine göre fark |
|---|---|---|
| Deniz testi (Prestero 7.5.1, yaklaşık) | 10.0 | – |
| robomodel, sadece CAD | 27.9 | +179% |
| robomodel, Prestero'nun sönüm düzeltmeleriyle (Tablo 8.2) | 8.1 | -19% |

Deniz testindeki değer bir grafikten okunmuş yaklaşık değerdir ("approximately 10
degrees/second"); ±%20 civarında belirsizlik kabul edilmeli.
