# 📊 Yazar-Makale İşbirliği Ağı Görselleştirme ve Analiz Sistemi

Bu proje, akademik bir veri setindeki (xlsx) yazar ve makale verilerini işleyerek, yazarlar arasındaki işbirliği ağını grafik teorisi prensiplerine göre görselleştiren web tabanlı bir analiz aracıdır.

Proje, yazarları "düğüm" (node), aralarındaki işbirliklerini ise "kenar" (edge) olarak modeller ve bu yapı üzerinde en kısa yol, en uzun yol ve ağaç yapıları gibi çeşitli algoritmaları çalıştırır.



## 🚀 Proje Hakkında

Bu uygulama, büyük veri setlerini işleyerek tarayıcı tabanlı interaktif bir arayüz sunar. Kullanıcılar, grafik üzerinde gezinerek yazarların işbirliği yoğunluğunu, birbirlerine olan uzaklıklarını ve kümeleşmelerini analiz edebilirler.

## Proje Katılımcıları
<a href="https://github.com/numankhy28">
  <img src="https://avatars.githubusercontent.com/u/219421415?v=4" width="128" height="128" alt="numankhy28 GitHub Profile" />
</a>
<a href="https://github.com/Furkanaydn44">
  <img src="https://avatars.githubusercontent.com/u/174454978?v=4" width="128" height="128" alt="Furkanaydn44 GitHub Profile" />
</a>


### Temel Özellikler

* **Veri İşleme ve Önbellekleme:** Excel formatındaki veriler okunur ve performans için RAM üzerinde önbelleğe (caching) alınır.
* **Dinamik Görselleştirme:** Yazarların (düğümlerin) boyutları, işbirliği sayılarına (derece/degree) göre dinamik olarak değişir.
* **İnteraktif Arayüz:** Tarayıcı üzerinden çalışan butonlar ve metin kutuları ile 7 farklı analiz fonksiyonu eş zamanlı olarak çalıştırılabilir.
* **Yazar/Makale Detayları:** Grafikteki düğümlerin üzerine gelindiğinde yazar ve makale bilgileri kayar yazı (tooltip) olarak gösterilir.

## 🛠️ Teknik Altyapı ve Mimari

Proje **Python** dili ile geliştirilmiş olup, web arayüzü için **HTML** ve **JavaScript** ile desteklenmiştir.

### Kullanılan Kütüphaneler

* **Backend & Server:**
    * `Dash`: Uygulamanın web arayüzünü ve sunucu yapısını oluşturur.
    * `Flask`: Dash kütüphanesinin arka planında sunucu görevini üstlenir.
* **Veri İşleme:**
    * `Pandas`: Excel (.xlsx) dosyalarının okunması ve veri manipülasyonu için kullanılır.
    * `Joblib`: İşlenen verilerin RAM üzerinde saklanması (serialization) için kullanılır.
* **Grafik ve Görselleştirme:**
    * `NetworkX`: Grafik yapısının (düğümler ve kenarlar) oluşturulması, konumlandırma ve algoritma hesaplamaları için kullanılır.
    * `Plotly`: Hesaplanan grafiğin tarayıcı üzerinde interaktif olarak çizdirilmesi için kullanılır.
    * `Matplotlib`: İkili Arama Ağacı (BST) gibi özel yapıların ayrı pencerelerde görselleştirilmesi için kullanılır.

## 🧮 Algoritmalar ve Analiz Modülleri

Proje içerisinde 7 temel ister (gereksinim) modülü bulunmaktadır:

### 1. En Kısa Yol Analizi (Shortest Path)
İki yazar (ID) arasındaki en kısa işbirliği yolunu hesaplar.
* **İşleyiş:** Seçilen iki düğüm arasındaki yol bulunur.
* **Görselleştirme:** Bu yol üzerindeki kenarlar **siyah renk** ile boyanır, kalınlaştırılır ve grafikte belirgin hale getirilir.



### 2. Ağırlıklı İşbirliği Kuyruğu
Belirli bir yazarın işbirliklerini ağırlıklarına göre analiz eder.
* **İşleyiş:** Girilen ID'ye sahip yazarın komşuları taranır ve kenar ağırlıklarına göre bir kuyruk yapısı oluşturulur.
* **Sonuç:** Ağırlıklara göre sıralanmış bir liste ve güncellenmiş grafik sunulur.

### 3. İkili Arama Ağacı (BST) Görselleştirmesi
İşbirliği ağındaki verileri kullanarak bir Binary Search Tree (BST) oluşturur.
* **İşleyiş:** 1. modülden elde edilen verilerle dengeli bir ağaç yapısı kurulur. İstenilen bir düğüm ağaçtan çıkarılabilir (delete node).
* **Görselleştirme:** Oluşturulan ağaç yapısı `Matplotlib` kullanılarak ayrı bir pencerede görselleştirilir.



### 4. İşbirliği Ağırlık Analizi
* Girilen ID'nin komşularını ve işbirliği ağırlıklarını analiz ederek listeler ve grafiği günceller.

### 5. Yazar İşbirliği Sayacı
* Girilen yazar ID'sine göre, o yazarın toplam işbirliği sayısını ve ismini veri setinden çekerek ekrana yazdırır.

### 6. En Çok İşbirliği Yapan Yazar (Max Degree)
* Tüm veri seti taranarak en fazla işbirliğine (en yüksek düğüm derecesine) sahip yazar tespit edilir ve bilgileri panele yazdırılır.

### 7. En Uzun Yol Analizi (Longest Path)
Bir yazardan başlayarak gidilebilecek en uzak mesafeyi hesaplar.
* **İşleyiş:** Verilen düğümden başlayarak tüm komşular taranır ve bir son (uç nokta) aranır.
* **Görselleştirme:** Bulunan en uzun yol, grafikte yeni bir kenar çizimi ile gösterilir.

## 🚀 Kurulum

1.  Depoyu klonlayın.
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install dash pandas networkx plotly joblib matplotlib openpyxl
    ```
3.  `app.py` (veya ana dosya) dosyasını çalıştırın.
4.  Tarayıcınızda `http://127.0.0.1:8050/` adresine giderek uygulamayı kullanın.

---
*Bu proje, grafik teorisi ve veri görselleştirme teknikleri kullanılarak geliştirilmiştir.*
