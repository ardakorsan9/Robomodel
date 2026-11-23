# BlueROV2 Heavy doğrulaması: robomodel ve von Benzon vd. (2022)

Bu dosya `validation/bluerov2.py` ile üretilir, elle düzenleme.

Kaynak: F. von Benzon vd., *An Open-Source Benchmark Simulator: Control of a BlueROV2
Underwater Robot*, J. Mar. Sci. Eng. 10 (2022) 1898, Tablo A1, 4, 5.
CAD: Blue Robotics'in açık BlueROV2 Heavy montajı (1142 parça, açık kasa).

Makalede **sönüm havuzda deneyle bulunmuş** (doğrulama hatası ~%10–15); eklenmiş kütle
ampirik (Eidsvik). Kütle, hacim, kütle merkezi ve eylemsizlik makaleden ölçülmüş değer
olarak girildi; karşılaştırmada sadece CAD'den hesapladıklarımız var.

## Geometri (CAD işleme)

| Büyüklük | robomodel | Makale | Fark |
|---|---|---|---|
| A_u (önden) | 0.0921 m² | 0.0877 m² | +5% |
| A_v (yandan) | 0.1060 m² | 0.1131 m² | -6% |
| A_w (üstten) | 0.2072 m² | 0.2049 m² | +1% |
| Hacim (CAD'den, voksel) | 0.0183 m³ | 0.0134 m³ (ölçülen) | +36% |

## Sönüm (makalede deneyle ölçülmüş)

| Eksen | robomodel karesel | Makale karesel | Fark | robomodel doğrusal | Makale doğrusal |
|---|---|---|---|---|---|
| ileri (u) | 50.6 | 141 | -64% | 5.06 | 13.7 |
| yan (v) | 58.3 | 217 | -73% | 5.83 | 0 |
| dikey (w) | 114 | 190 | -40% | 11.4 | 33 |
| yuvarlanma (p) | 0.598 | 1.19 | -50% | 0.12 | 0 |
| yunuslama (q) | 0.295 | 0.47 | -37% | 0.0591 | 0.8 |
| sapma (r) | 0.323 | 1.5 | -78% | 0.0647 | 0 |

Birimler: karesel N s²/m² (dönmede N m s²/rad²), doğrusal N s/m (N m s/rad).
Makalenin doğrusal terimleri ölçülmüş; bizimki karesel × 0,1 m/s varsayımı.

## Eklenmiş kütle (makalede de ampirik tahmin)

| Eksen | robomodel (BEM) | Makale (Eidsvik) | Fark |
|---|---|---|---|
| ileri (u) | 9.04 | 6.36 | +42% |
| yan (v) | 14.4 | 7.12 | +103% |
| dikey (w) | 36.1 | 18.7 | +93% |
| yuvarlanma (p) | 0.432 | 0.189 | +128% |
| yunuslama (q) | 0.282 | 0.135 | +109% |
| sapma (r) | 0.266 | 0.222 | +20% |

İkisi de tahmin; hangisinin doğru olduğunu bu veriyle bilemeyiz.

## Davranış: 85 N ileri itkide son hız

| Model | Son hız (m/s) | Makaleye göre |
|---|---|---|
| Makale (havuzda ölçülen sönümle, Tablo 4: 0.72 m/s) | 0.73 | – |
| robomodel, sadece CAD | 1.25 | +71% |

## Yorum

- CAD işleme iyi: izdüşüm alanları makalenin CAD değerleriyle birkaç yüzde içinde.
- Hacim CAD'den iyi çıkmıyor (vokseller ince plakaları kalınlaştırıyor, tüplerin
  içindeki hava CAD'de parça olarak yok): açık kasalı araçta suda tartmak şart.
- Sönüm REMUS'taki gibi CAD'den düşük çıkıyor: C_D = 1,1 ile 1,6–4,6 kat az. Makalenin
  değerleri izdüşüm alanına göre C_D ≈ 1,9 (dikey) – 3,8 (yan) demek; iticiler, kablolar
  ve kasa içinden geçen akış bunu büyütüyor. Doğru değer için havuz testi ve
  `robomodel calibrate` gerekli.
