# History
I denne filen kan man skrive historie/logg hvis man har lyst. Bilder er kult.

### [19.09.2020]
Funnet ut hvordan man kan bruke se fremover flere enn ett skritt. Nå ser jeg fremover 15 skritt basert på  
de 30 bak. Da gir det en kurve som har ser ganske grei ut med tanke på at det kan være mye feil.
<img src="../images/RNN_15_ahead.png" alt="df"/><br>

### [18.09.2020]
Funnet ut hvordan jeg kan ha flere feauters (x) inn i det trente nettverket.  
Trente på en runde og fikk ganske godt resultat når jeg kjørte prediksjonen på andre testsett. 
<img src="../images/RNN_on_untrained.png" alt="df"/><br><br>

### [18.09.2020]
Normalisert ydata for å få en spredning som er lettere å lære for RNN.  
Trener på den deriverte 30sma for å standardisere prisvekst.  
<img src="../images/RNN_on_derivert30sma.png" alt="df"/><br>

### [16.09.2020]
Jeg er ikke helt overbevist over standardavvik-tilnærming, så tester jeg ut maskinlæringsmodeller.  
Først tester jeg RNN med LSTM. Her tester jeg og trene på samme datasett bare for å vise at det kan funke.  
Det andre bildet er kun basert på RSI. Gir noe utslag i slutten som er bra.  
<img src="../images/RNN_on_price.png" alt="df"/><br>
<img src="../images/RNN_on_RSI.png" alt="df"/><br>

### [30.08.2020]
Satt sammen første trade plot basert på modellen. Oppgangen er der den vil holde posisjon og ned er der den vil være ute.<br>
Det var sånn passe vellykket. Det virker som den ikke helt vil legge inn posisjon i oppganene men kun når det er flatt..<br>
<img src="../images/firsttradeplot.png" alt="df"/><br>

### [28.08.2020]
Funnet mean og stdv for hver oppgang på de forskjellige indikatorene. Dataframenen ser fint ut:<br>
<img src="../images/df_statistikk.PNG" alt="df"/><br>
Her er trix med stdv for å avgrense hvor man kan ligge innenfor. Nå regnes det med 2 stdv for å få ca. 95% av innenfor.<br>
Kan se om jeg finner noen statistikktabeller for å finne ut av nøyaktig z*.<br>
<img src="../images/trix_med_stdv.png" alt="df"/><br>

### [26.08.2020]
Skyter litt i blinde for å se etter gode indikatorer, så ønsker å prøve å lage en modell basert på det som ser bra nok ut til nå.<br>
De som har gjort det bra er: trix, rsi, sma16-8 og macds.<br>
Neste mål blir å lage en modell ut av disse<br>
<img src="../images/trixplot_2.png" alt="trix" width="270"/>
<img src="../images/rsiplot_2.png" alt="rsi" width="270"/>
<img src="../images/sma8-16plot_2.png" alt="sma" width="270"/>
<img src="../images/macdsplot_2.png" alt="macds" width="270"/>

### [24.08.2020]
Sett over alle 80 filer/dager og hentet ut ca 35 gode løp med over 1 prosent stigning og score over 30.<br>
Tatt å hentet ut fem indikatorer: trix, rsi, adxr, kdjk og pdi for å se hva de har å by på.<br>
Resultatet og mønsteret ligger under. Kun trix har øyeblikkelig gjenkjennelig mønster.<br>
<img src="../images/trixplot.png" alt="trix" width="220"/>
<img src="../images/rsiplot.png" alt="rsi" width="220"/>
<img src="../images/adxrplot.png" alt="adxr" width="220"/>
<img src="../images/kdjkplot.png" alt="kdjk" width="220"/>
<img src="../images/pdiplot.png" alt="pdi" width="220"/>

### [20.08.2020]
Lagd fine grafer som gir en bilde av når oppgangen starter og slutter. Grønn er start, rød er slutt.
![eksempelbilde2](../images/eksempel2.png)

### [19.08.2020]
Nå har jeg funnet ut en grei måte å få til å se når det er positiv vekst i grafen.<br>
Basert på den akumelerte scoren av positiv utvikling og prosentvis oppgang kan jeg se hvilke oppganger jeg har lyst til å ta vare på<br>
Neste mål blir å koble indikatorer til oppgangene.<br>
<img src="../images/posutvikling.png" width="400"/>
<img src="../images/posutviklingpris.png" width="400"/>

### [06.08.2020]
Start project. Lastet opp fra colab og organisert.