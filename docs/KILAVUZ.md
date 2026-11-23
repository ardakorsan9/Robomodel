# robomodel kullanım kılavuzu

CAD çizimi yaptın ve MATLAB'da aracın matematiksel modelini kontrol etmek istiyorsun. Bu sayfa
baştan sona ne yapacağını anlatıyor.

## 1. Kurulum (bir kez)

Python 3.12 gerekir. Proje klasöründe:

```bash
uv sync
```

(`uv` yoksa: `pip install -e .`)

## 2. CAD'i hazırla

- Fusion 360 / SolidWorks'ten **STL** (tercihen) ya da **STEP** olarak dışa aktar.
- Tek parça kapalı gövde en iyisi. Çok parçalı açık kasa (ROV) da olur: "assembly" seçeneği.
- Birimi bil (genelde mm).

### Kütle nereye düşüyor?

Terazi toplam kütleyi verir, kütlenin nerede olduğunu vermez. CAD'i **STEP montajı** olarak dışa
aktarırsan robomodel her parçayı ayrı tutar. Parçanın adından malzemesini tahmin eder (örneğin
"LiPo" → batarya, "screw/vida" → çelik, "foam/köpük" → köpük), parça kütlelerini hesaplar ve
hepsini tarttığın toplam kütleye ölçekler. Senin ek bir şey girmen gerekmez. Parçalara anlamlı
adlar vermek ("battery", "motor_mount" gibi) sonucu iyileştirir. Tek parça STL'de kütle eşit
dağıtılır. En doğrusu yine ağır parçaları listelemek ya da ağırlık merkezini ölçmektir.

## 3. En kolay yol: web arayüzü

```bash
uv run robomodel serve
```

Tarayıcıda http://127.0.0.1:8000 aç:

1. **CAD dosyasını seç.** 3B önizleme açılır.
2. **Eksenleri kontrol et.** Kırmızı ok aracın önünü, mavi ok aşağıyı göstermeli. Göstermiyorsa
   "Forward is file axis" ve "Up is file axis" seçimlerini değiştir.
3. **Kütleyi gir** (tartıdaki değer).
4. **İticileri yerleştir:** "+ thruster"a bas, modelin üzerinde iticinin olduğu yere tıkla, itme
   yönünü seç. T200 ise model "T200" kalsın; değilse itki sınırlarını (N) yaz.
5. Varsa **kanatları** ve **ağır iç parçaları** (batarya) aynı şekilde ekle.
6. **Build model** → kontroller ve rapor görünür → **Download** ile zip'i indir.

## 4. MATLAB'da kullan

Zip'teki `matlab` klasörüne geç:

```matlab
arac_example              % simülasyon: orta, düşük ve yüksek model birlikte
p = arac_params();        % en iyi tahmin
lin = load('arac_linear.mat');  K = lqr(lin.A, lin.B, Q, R);
arac_build_simulink       % Simulink modeli: arac_sim.slx (giriş tau, çıkış durum)
```

**Üç model neden var?** Su direnci (sönüm) CAD'den kesin çıkmaz: test ettiğimiz iki araçta
gerçek değer CAD tahmininin 1,6–10 katıydı. Kontrolcünü `_params` (orta) ile tasarla, sonra
`_params_low` (az sönüm, az eklenmiş kütle, az kanat kaldırması) ve `_params_high` (çok sönüm,
çok eklenmiş kütle) ile de dene. Bu iki uç her davranışta "hızlı" ve "yavaş" demek değil (örneğin
az kanat kaldırması aracı daha az kararlı yapar), ama gerçek araç büyük ihtimalle aralarında.
Kontrolcü üçünde de çalışıyorsa gerçek araçta da büyük ihtimalle çalışır.

## 5. Raporu nasıl okumalı?

- **Checks:** `ok` iyi. `WARN` dikkat: örneğin "self-righting" uyarısı, aracın devrilince kendi
  kendine düzelmeyeceği demek (ağırlık merkezini aşağı al). `FAIL` model kullanılamaz: girişte
  hata var.
- Her değerin yanında **± belirsizlik** ve **kaynak** var: `measured` (ölçüldü), `cad`,
  `analytic` (kesin formül), `empirical` (el kitabı tahmini), `assumed` (varsayım).

## 6. Daha doğru model için (araç varsa)

| Ne | Kazanç | Süre |
|---|---|---|
| Suda tart → `measured.volume` | Kaldırma kuvveti %0,5 doğrulukta | 10 dk |
| Denge testi → `measured.r_g` | Kararlılık doğru | 20 dk |
| Havuz testi + `robomodel calibrate` | Hız, dönüş, frenleme %10–20'ye iner | 1 saat |

Hangisinin önce gerektiğini `uv run robomodel wizard arac.yaml` söyler. Havuz testinin planı:
`uv run robomodel testplan arac.yaml -o plan`.

## 7. Su üstü aracı (tekne, USV)

Web sayfasında ilk adımda **Su üstü aracı**'nı seç (YAML'da `category: surface`). Gövde
**güverte dahil kapalı** olmalı, çünkü su hattı CAD'den bulunuyor. Raporda şunları görürsün:

- **Su çekimi, trim, yatma:** tekne suda nasıl duruyor.
- **GM_T:** yan yatma dengesi. Pozitifse tekne kendini doğrultur, negatifse devrilir.
- **GZ eğrisi:** büyük açılarda doğrultma kolu ve kaç dereceye kadar doğrulduğu.
- **Doğal periyotlar:** dikey salınım, yatma ve trim.

Bu sonuçların hepsi **ağırlık merkezinin yüksekliğine** bağlı. Batarya ve motor gibi ağır
parçaları listele ya da ağırlık merkezini ölç (`measured.r_g`). MATLAB çıktısında
`p.surface = true`, `p.G` ve `p.roll_table` bulunur, simülasyon tekne yüzerken başlar. Gazebo
çıktısı su üstü araçları için henüz yok. Doğruluk raporu: `docs/dogrulama_su_ustu.md`.

## 8. Drone (çok rotorlu)

Web sayfasında **Hava aracı**'nı seç (YAML'da `category: aerial`). 3. adımda **+ rotor** ile her
motorun yerine tıkla. Her rotor için şunları gir:

- dönüş yönü (üstten bakınca cw / ccw),
- azami itki (N; itki standından ya da motor tablosundan),
- pervane çapı (inç),
- motor + pervane kütlesi (g).

Bataryayı **ağır parça** olarak ekle. Rapor, drone'un düz şekilde havada asılı kalıp
kalamayacağını, hover gazını, itki/ağırlık oranını ve rotorların dört yönü (itki, yatış,
yunuslama, dönüş) kontrol edip etmediğini söyler. MATLAB örneği hover gazıyla başlar ve drone
yerinde durur; kontrolcünü bunun üstüne kurarsın. Doğruluk raporu: `docs/dogrulama_drone.md`.

## 9. Kara robotu (tekerlekli, paletli)

Web sayfasında **Kara robotu**'nu seç (YAML'da `category: ground`) ve zemini belirle (beton, iç
mekân, asfalt, çakıl, çim, kum). 3. adımda **+ tekerlek / palet** ile her tekerleğin
**göbeğine** tıkla. Her tekerlek için şunları gir:

- yarıçap ve genişlik,
- tekerlek + motor kütlesi,
- tahrikli mi, yönlendirmeli mi, sarhoş teker mi,
- tahrikliyse motor değerleri: tekerlekte, yani dişli kutusundan sonra durma torku (N·m) ve
  yüksüz hız (rpm).

Rapor şunları söyler:
- robot tekerlekleri üstünde duruyor mu,
- kaç derecede devrilir,
- son hız ve azami ivme,
- çıkabileceği en dik eğim,
- dönüş hızı ya da en dar dönüş yarıçapı.

MATLAB çıktısında `robomodel_ground_dynamics.m` var: girdi `u = [tahrik komutları; direksiyon
açıları]`, durum `[x y ψ u v r]`. Doğruluk raporu: `docs/dogrulama_kara.md`.

## 10. Robot kol (eklemli)

Web sayfasında **Robot kol**'u seç.

1. **CAD:** her link için bir dosya seç, birkaçını aynı anda da seçebilirsin. Parçaları
   montajdan **sıfır pozda ve aynı koordinatlarda** dışa aktar. Fusion 360 ve SolidWorks
   gövdeleri tek tek dışa aktarırken bunu kendiliğinden yapar. İlk dosya sabit tabandır.
2. **Kütle:** her linkin motoru ve dişli kutusuyla birlikte tartılmış kütlesini gir.
3. **Eklemler:** **+ eklem** ile her eklemin dönme ekseninin üstüne tıkla. Sonra şunları seç:
   tip (dönel, sürekli, kayar, sabit), bağlı olduğu link, hareket eden link, eksen (z yukarı),
   sınırlar ve motor sınırı.

Sonuçta şunlar var:
- URDF ve meshler (ROS, Gazebo, MATLAB),
- kontroller,
- her motorun kolu yerçekimine karşı tutmak için gereken tork: çizilen pozda ve eklem
  sınırları içindeki en kötü pozda. Motor seçerken buna ivme ve taşınacak yük için pay ekle.

**MATLAB için toolbox gerekmez:** `robomodel_robot_mass.m` M(q)'yu, `robomodel_robot_bias.m`
Coriolis + yerçekimi + sürtünmeyi, `robomodel_robot_dynamics.m` ise ẋ = f(x, τ)'yu verir
(ode45 ya da Simulink'te MATLAB Function bloğu için). `matlab/<ad>_example.m` bunları
Python'la karşılaştırır: M(q) farkı 4·10⁻¹⁵, 2 s'lik salınım farkı 9·10⁻⁹ rad. Robotics System
Toolbox varsa `<ad>_check_toolbox.m` aynı kontrolü `importrobot` ile yapar.

## 11. Gazebo (ROS 2)

```bash
uv run robomodel gazebo arac.yaml -o gz
cd gz && GZ_SIM_RESOURCE_PATH=$PWD gz sim -r world.sdf
```

İtici komutu (N): `gz topic -t /model/arac/joint/thruster0_joint/cmd_thrust -m gz.msgs.Double -p 'data: 10'`

## Sık hatalar

| Belirti | Sebep | Çözüm |
|---|---|---|
| "not closed (watertight)" | CAD'de delik ya da çok parça | "assembly" seç ya da tek katı gövde aktar |
| "longest along body z" | Eksenler yanlış | Önizlemede ön/üst seçimini düzelt |
| Hacim çok büyük/küçük | Birim yanlış (mm/m) | Birim seçimini düzelt |
| "floats up / sinks" | Kütle ile hacim uyuşmuyor | Kütleyi tart, hacmi suda tartarak gir |
