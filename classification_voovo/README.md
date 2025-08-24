# Active Learning Document Classifier

Ez a projekt egy intelligens dokumentum-oszt�lyoz� rendszer, amely az akt�v tanul�s (Active Learning) m�dszer�t alkalmazza. A c�lja, hogy k�l�nb�z� form�tum� (PDF, DOCX, PPTX) dokumentumokb�l kinyert sz�veges bekezd�seket automatikusan, el�re defini�lt t�m�k szerint kategoriz�lja, mik�zben minimaliz�lja a k�lts�ges API h�v�sok sz�m�t.

## F�bb Jellemz�k

-   **Akt�v Tanul�si Ciklus:** A rendszer egy gyors, helyi "di�k" (Student) neur�lis h�l�t �s egy nagy teljes�tm�ny� "or�kulum" (Oracle - Google Gemini API) modellt haszn�l. Csak akkor fordul a dr�ga or�kulumhoz, ha a di�k modell bizonytalan, majd az �gy kapott �j tud�ssal dinamikusan �jratan�tja a di�kot.
-   **Dinamikus �jratan�t�s:** A di�k modell a fut�s k�zben, 25-�s csomagokban (`RETRAIN_TRIGGER_COUNT`) gy�jt�tt �j adatokb�l automatikusan �jratanul, �gy folyamatosan okosabb� �s magabiztosabb� v�lik.
-   **P�rhuzamos API H�v�sok:** Az aszinkron feldolgoz�snak k�sz�nhet�en a rendszer k�pes egyszerre t�bb API k�r�st is elk�ldeni, jelent�sen cs�kkentve a v�rakoz�si id�t.
-   **Univerz�lis Dokumentumfeldolgoz�s:** K�pes `.pdf`, `.docx`, �s `.pptx` f�jlok sz�veges tartalm�nak kinyer�s�re �s intelligens bekezd�sekre bont�s�ra.
-   **K�zpontos�tott Konfigur�ci�:** Minden fontos param�ter (modellnevek, el�r�si utak, k�sz�b�rt�kek) egyetlen helyen, a `config/constants.py` f�jlban m�dos�that�.

## A Rendszer M�k�d�se

A projekt lelke a "di�k-or�kulum" modell. A di�k egy kicsi, gyors neur�lis h�l�, ami a helyi g�pen fut. Az or�kulum a nagy teljes�tm�ny� Gemini Pro modell. A c�l, hogy a di�k a lehet� legt�bb munk�t elv�gezze, �s csak akkor k�rjen seg�ts�get a "b�lcs" or�kulumt�l, ha abszol�t sz�ks�ges. A folyamat a k�vetkez�k�ppen n�z ki:

### Folyamat�bra

*(Megjegyz�s: A lenti diagram megjelen�t�s�hez olyan Markdown-megjelen�t� sz�ks�ges, amely t�mogatja a Mermaid.js szintaxist, mint p�ld�ul a GitHub.)*

```mermaid
graph TD
    A[Start] --> B[Dokumentumok beolvas�sa, bekezd�sek kinyer�se];
    B --> C{For each Paragraph};
    C --> D{Benne van az Oracle Cache-ben?};
    D -- Igen --> E[Gyors�t�t�razott pontsz�mok haszn�lata];
    E --> Z[K�vetkez� bekezd�s];

    D -- Nem --> F{L�tezik Student modell?};
    F -- Nem --> G[Hozz�ad�s az Oracle V�r�list�hoz];

    F -- Igen --> H[Predikci� a Student modellel];
    H --> I{"A Student magabiztos? (>= 0.85)"};
    I -- Igen --> J[Student pontsz�mainak haszn�lata];
    J --> Z;
    I -- Nem --> G;

    G --> K{"V�r�lista m�rete >= 25?"};
    K -- Nem --> Z;

    K -- Igen --> L;
    
    subgraph L [Batch Feldolgoz�s �s Tan�t�s]
        M[P�rhuzamos API h�v�s a 25 bekezd�sre]
        M --> N[�j c�mk�k ment�se a Cache-be]
        N --> O[Teljes Cache bet�lt�se]
        O --> P[Student Modell �jratan�t�sa]
        P --> Q[�j modell bet�lt�se a mem�ri�ba]
    end

    L --> R[V�r�lista ki�r�t�se];
    R --> Z;

    C -- Nincs t�bb bekezd�s --> S{Maradt valami a V�r�list�n?};
    S -- Igen --> T[Marad�k Batch Feldolgoz�sa �s Tan�t�s];
    T --> U[V�geredm�nyek ment�se JSON-be];
    S -- Nem --> U;
    U --> V[End];
```

**BEFORE THE FIRST RUN:**
To run the code, you need to install the required libraries. You can do this by running the following command in your terminal or command prompt:
```bash
pip install pdfplumber python-docx python-pptx tensorflow scikit-learn scipy
```