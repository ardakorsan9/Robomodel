# robomodel

Robotun CAD dosyasından ve kısa bir araç tanımından matematiksel modelini (Fossen 6 serbestlik
dereceli) çıkaran yazılım. İlk hedef sualtı araçları. Mimari sonradan rover ve drone gibi
kategorilere genişletilebilecek şekilde kuruldu.

Her parametre değeri, belirsizliği ve kaynağıyla (ölçülmüş, kalibre edilmiş, BEM, ampirik)
birlikte tutulacak. Böylece rapor hangi değere güvenilebileceğini açıkça söyler.

## Kurulum

İlk seferde, proje klasöründe:

```bash
cd ~/projects/robomodel && uv sync
```

## Kullanım

Araç tanım dosyasını kontrol et:

```bash
uv run robomodel check examples/sphere.yaml
```

Modeli hesapla ve raporu göster (`-o` ile tüm parametreleri YAML olarak kaydeder):

```bash
uv run robomodel build examples/test_auv.yaml -o test_auv_model.yaml
```

MATLAB dosyalarını üret (`--speed`: doğrusal modelin çalışma noktasındaki ileri hız, m/s):

```bash
uv run robomodel matlab examples/test_auv.yaml -o ~/veri/kayitlar/robomodel/test_auv_matlab --speed 1.0
```

MATLAB'da o klasöre geçip `test_auv_example` çalıştır. Simülasyonu yapar, Python sonucuyla
karşılaştırır ve grafiği çizer. Kontrolcü tasarımı için `test_auv_linear.mat` içindeki
`A`, `B` matrislerini kullan.

Testleri çalıştır:

```bash
uv run pytest
```

## Dosyalar

| Dosya | İş |
|---|---|
| `src/robomodel/vehicle.py` | Araç tanım dosyasını (YAML) okur ve doğrular |
| `src/robomodel/parameter.py` | `Parameter`: değer, belirsizlik, kaynak |
| `src/robomodel/geometry.py` | Küre, elipsoit, silindir, Myring ağları; STL/OBJ okuma |
| `src/robomodel/rigid_body.py` | Kütle merkezi, eylemsizlik, M_RB |
| `src/robomodel/hydrostatics.py` | Hacim, kaldırma merkezi, W, B, g(η) |
| `src/robomodel/sections.py` | Gövdeyi x boyunca dilimler (genişlik, yükseklik, alan) |
| `src/robomodel/added_mass.py` | Eklenmiş kütle: elipsoitte Lamb (kesin), diğer gövdelerde şerit teorisi |
| `src/robomodel/damping.py` | Sönümleme tahmini (Hoerner): doğrusal + karesel |
| `src/robomodel/thrusters.py` | İtici eğrisi ve dağıtım matrisi B |
| `src/robomodel/dynamics.py` | Fossen 6-DOF denklemleri, simülatör, doğrusallaştırma |
| `src/robomodel/matlab_export.py` | MATLAB dosyaları |
| `src/robomodel/model.py` | Hepsini birleştirir, rapor ve YAML çıktısı |
| `src/robomodel/cli.py` | Komut satırı (`check`, `build`, `matlab`) |
| `examples/*.yaml` | Küre, elipsoit (L/D = 5), REMUS gövdesi, basit AUV |
| `tests/test_physics.py` | Geometri, kütle, hidrostatik, Lamb: analitik cevaplara karşı |
| `tests/test_dynamics.py` | Şerit teorisi, sönümleme, iticiler, simülatör, MATLAB çıktısı |
| `docs/eksenler.md` | Eksen ve işaret kuralları. Kod yazmadan önce oku |
| `docs/plan.md` | Fazlar, saat tahminleri, doğrulama merdiveni |

## Araç tanım dosyası

```yaml
name: test_auv
category: underwater        # şimdilik tek kategori
mass: 31.0                  # kg, toplam
water: fresh                # sea (1025), fresh (1000) ya da kg/m^3
geometry:
  primitive: cylinder       # sphere, ellipsoid, cylinder, myring ya da file: govde.stl
  length: 1.6
  radius: 0.095
parts:                      # isteğe bağlı: ağır iç parçalar
  # shape: point (varsayılan), sphere (radius), box (size), cylinder (radius, length, axis)
  - {name: battery, mass: 6.0, position: [0.1, 0.0, 0.04], shape: box, size: [0.3, 0.1, 0.06]}
thrusters:
  # T200 için itki sınırları hazır (kontrol edilmeli); başka modelde max_forward/max_reverse ver (N)
  - {model: T200, position: [-0.85, 0.0, 0.0], direction: [1, 0, 0]}
```

Konumlar gövde ekseninde, metre cinsinden: x ileri, y sağ, z aşağı.
