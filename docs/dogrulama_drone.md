# Doğrulama: çok rotorlu drone

Testler: `tests/test_aerial.py` · Örnek: `examples/quad.yaml`

## Model

`M_RB ν̇ + C_RB(ν) ν + D(ν) ν + g(η) = B f`

- **M_RB:** kütle, ağırlık merkezi (CG) ve atalet. CAD gövdesi ile parça listesinden (batarya,
  `motor_mass`) hesaplanıyor. Ek kütle ve kaldırma kuvveti yok: hava, sudan ~800 kat hafif.
- **g(η):** CG'ye etkiyen yerçekimi.
- **B (rotor dağıtım matrisi):** her rotor için `[a; r × a + κ a]`. Burada a itki yönü, r rotor
  konumu, κ tork/itki oranı (cw için +, ccw için −).
- **İtki:** f = f_max · komut².
- **Sönüm:**
  - Şasi sürüklemesi: izdüşüm alanlarında C_D = 1,1, ROV yöntemiyle, hava yoğunluğuyla.
  - Rotor sürüklemesi: momentum teorisindeki "ram drag" ṁ·v'nin yarısı, ṁ = √(ρ A T / 2)
    (varsayım).
- **Rüzgâr:** `wind: [kuzey, doğu, aşağı]` m/s. Su akıntısıyla aynı bağıl hız biçimi kullanılıyor.

## Kontroller (her modelde)

| Kontrol | Ne bakıyor |
|---|---|
| Düz hover mümkün mü | 0 ≤ f ≤ f_max sınırları içinde yerçekimini ve CG momentini dengeleyen bir itki var mı |
| İtki/ağırlık ≥ 1,5 | Kontrol ve rüzgâr için yeterli yedek itki |
| Rotorlar 4 serbestliği kontrol ediyor mu | B'nin itki, yatış, yunuslama, dönüş satırlarının rankı 4 mü |
| cw/ccw dengesi | Çift sayıda rotorda dönüş yönleri eşit mi (uyarı) |

## Analitik doğrulamalar (testler)

| Durum | Beklenen | Sonuç |
|---|---|---|
| İşaretler: ön rotor burnu kaldırır, sağ rotor sola yatırır, cw rotor gövdeyi ccw döndürür | el hesabı | tam |
| Simetrik + quad hover'da itki | her rotor W/4, komut √(W/4/f_max) | 10⁻⁶ |
| Hover komutuyla 5 s simülasyon | yerinde kalır | 10⁻⁶ m |
| CG 5 cm önde | moment dengesi (f_ön − f_arka)·kol = W·x_g | 10⁻⁶ |
| CG rotorların dışında | hover yok, kontrol FAIL | doğru |
| Serbest düşüş | ilk anda g ivmesi; sonunda sürükleme limit hızı | %1 / 10⁻⁴ |
| Rotor sürüklemesi | 4 · 0,5 · ṁ, rotor ekseni boyunca sıfır | 10⁻⁶ |
| 60 g'lık 4 motor ekleyince yaw ataleti | 4 m (x² + y²) − gövdenin kaybettiği kütlenin payı | 10⁻⁶ |

## Neye güvenilir?

- **Güvenilir:** hover itkileri ve gaz, kontrol edilebilirlik, dağıtım matrisi B, kütle ve CG.
  Atalet de güvenilir, **yeter ki ağır parçalar (batarya, motorlar) listelensin**. Liste
  yoksa kütle CAD hacmine yayılıyor ve atalet yanlış yerde kalıyor (uyarı veriliyor).
- **Tahmin (ölçülmeli):**
  - κ tork/itki oranı: çaptan κ ≈ 0,065·D (C_T 0,11, C_P 0,045 varsayımı). Pervaneye göre
    2 kat değişebilir; itki standında tork ölçülürse `torque_ratio` ile girilir.
  - Rotor ve şasi sürüklemesi.
- **Modelde yok:** itkinin hava hızıyla düşmesi (bu yüzden tırmanma ve son hız yüksek çıkıyor),
  motor tepki süresi, jiroskopik rotor etkileri, yer etkisi, pervane kanat çırpması.
- **Gazebo çıktısı** drone için henüz yok (gz-sim'in MulticopterMotorModel eklentisine
  aktarım planlanıyor). MATLAB ve Simulink çıktıları çalışıyor.

## Gerçek veriyle karşılaştırma: yapılmadı

Bunun için CAD'i açık, parametreleri bir yayında ölçülmüş bir drone gerekiyor. Aday:
Crazyflie 2.x. CAD'i üreticinin sitesinde açık; kütle, atalet, κ ve itki eğrisi bir sistem
tanımlama tezinde ölçülmüş. Bu dosyalar indirilince REMUS ve BlueROV2'deki gibi bir
karşılaştırma yapılabilir.
