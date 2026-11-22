# Ürün durumu ve eksikler (2026-11-22)

Bu liste denemelere dayanıyor: tuhaf şekillerle stres testi, web API testleri, MATLAB,
Simulink ve Gazebo'da çalıştırma, REMUS 100 ve BlueROV2 karşılaştırmaları.

## Hazır olanlar

| Alan | Durum |
|---|---|
| CAD girişi | STL, STEP, OBJ, PLY; birim, eksen, serbest döndürme (`rotate`) |
| Bozuk CAD | Çift yüzey, dejenere üçgen, ters normal, delik (kapak kapatma), iç içe katı (voksel birleşimi), açık montaj (voksel) otomatik onarılıyor ve raporlanıyor |
| Tuhaf şekiller | Torus, ince plaka, katamaran, gürültülü tarama, 1 cm–20 m, eğik çizim, uzak orijin: 17 şekilde stres testi |
| Fizik | M_RB, M_A, C, D, g(η), kanat, gövde kaldırması, itici (ölü bölge, doyma), akıntı; hidrodinamik terimler gövde merkezinde hesaplanıp orijine taşınıyor (CAD orijininden bağımsız) |
| Güvenilirlik | Her değer ± belirsizlik ve kaynakla; alt/orta/üst model; 10 otomatik matematik kontrolü; birim ve yoğunluk makullük uyarısı |
| Çıktı | MATLAB (3 model, doğrusal A-B), Simulink kurucu, Gazebo (doğrulandı), YAML, rapor, sürüm ve girdi özeti |
| Web | Yükle, 3B önizleme, boyut gösterimi, tıklayarak itici/kanat/parça, örnek araç, sonuç paneli, zip, ayrı işlemde hesap (süre sınırı, eşzamanlı iş sınırı), 2 saatte otomatik silme, "şimdi sil", güvenlik başlıkları, sağlık kontrolü |
| Yayın | Dockerfile, yayın kılavuzu (`docs/DEPLOY.md`) |

## Eksikler (öncelik sırasıyla)

### A. Yayından önce şart

| # | Eksik | Neden | Tahmini süre |
|---|---|---|---|
| 1 | **Arayüzü gerçek tarayıcılarda uçtan uca denemek** (Chrome, Firefox, Safari, telefon) | API testli, ama tıklama ve 3B önizleme insan gözüyle denenmedi | 2–3 saat |
| 2 | **Lisans seçimi** (açık kaynak mı, ticari mi) | Yayınlamadan önce karar gerekir | Senin kararın |
| 3 | **Gizlilik politikası ve kullanım koşulları** (AB'deysen KVKK/GDPR) | CAD dosyaları ticari sır olabilir; doğruluk sınırları yazılmalı | 2–4 saat + hukuki kontrol |
| 4 | **Sunucu, alan adı, HTTPS, hız sınırı** | Docker hazır, ama denenmedi (bu bilgisayarda Docker yok) | 3–5 saat |
| 5 | **Windows ve macOS'ta kurulum denemesi** | Komut satırı kullanıcıları için | 1–2 saat |

### B. Doğruluğu artıran (CAD'den hâlâ çıkarılabilecekler)

| # | Eksik | Kazanç | Tahmini süre |
|---|---|---|---|
| 6 | **Capytaine (BEM) ile eklenmiş kütle** | Her şekilde eklenmiş kütle ±%5–10 (şimdi şerit teorisi ±%15–35) | 8–12 saat |
| 7 | **Çok gövdeli hidrodinamik** (katamaran, ayrık parçalar) | Şimdi tek gövde gibi hesaplanıp uyarılıyor | 6–8 saat |
| 8 | Büyük tek parça gövdelerde hız (1,3 milyon üçgen: 25 s) | Web için daha hızlı yanıt | 3–4 saat |
| 9 | İtici dış hacmi ve itici–gövde etkileşimi | Küçük araçlarda birkaç % | 4–6 saat |

### C. CAD'den çıkmayan (veri gerekir)

| # | Eksik | Yol |
|---|---|---|
| 10 | **Sönüm** (en büyük belirsizlik) | Havuz testi + `robomodel calibrate`; o zamana kadar alt/üst model |
| 11 | Gerçek araçla kalibrasyon kanıtı | En az bir aracı havuzda test etmek |
| 12 | Daha fazla doğrulama aracı | 3–5 araç daha → sönüm için istatistiksel düzeltme |

### D. Ürünleşme

| # | Eksik | Not |
|---|---|---|
| 13 | Türkçe/İngilizce arayüz dili seçimi | Arayüz şimdilik İngilizce |
| 14 | Proje kaydetme / geri yükleme (vehicle.yaml'ı arayüze geri yükleme) | Zip içinde vehicle.yaml var, arayüze geri yükleme yok |
| 15 | Kullanıcı hesapları, ödeme | İş modeline bağlı |
| 16 | Gazebo'ya akıntı ve dümen komutları | Şimdi dümen kuvvetleri pasif |
| 17 | Diğer robot türleri (rover, drone, kol) | Çekirdek hazır; ortam modülleri yazılacak |
