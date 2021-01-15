# History
I denne filen kan man skrive historie/logg hvis man har lyst. Bilder er kult.

### [15.01.2021]
Satt opp mulighet til å se grafisk hvor det handler. Etter 10 episoder handler den 70-90 handler og det er aaalt for mye.  
Her ser man hvordan det kan se ut. Grønn er kjøp og rød er salg  
<img src="../images/rl_mange_handler.png" width="500" alt="df"/><br>

### [14.01.2021]
I juleferien har jeg sett på Reinformance learning og har troa på at det er mulig å trene opp en agent til å handle.  
Satt opp et miljø (environment) som er mulig å bruke et deep q nettverk på. Tar ca 1 minutt å trene en aksjedag, så tar litt lang tid.

### [16.12.2020]
En ide som kom fram var å trene på formiddag og predikere på ettermiddagen. Siden forskjellen mellom hver aksje var så stor  
kunne det være at de forskjellige mønsterene å i aksjene. Etter å ha prøvd - kun på en aksje riktignok - viste det seg at  
r2 scoren (~0.32) ikke ble noe spesielt bedre og konkluderer med at det ikke helt er det store.
<img src="../images/treneformiddag.png" width="500" alt="df"/><br>

### [06.10.2020]
Lagt til akumulerte summen av den deriverte git kjøpsforslag. Et av de bedre for å se hvordan det funker på en  
god stigende trend.  
Lagt ved dataframen under slik man ser nøyaktig gevinst.  
<img src="../images/RNN_w_df.PNG" width="500" alt="df"/><br>

### [03.10.2020]
Lagt til kjøpsforslag basert på RNN. Funker bra.<br>
<img src="../images/RNN_buy_forslag.png" alt="df"/><br>

### [02.10.2020]
Klart å gjennomsnittet av prediksjonene. Det var vanskeligere enn først antatt.  
Liten tanke er at jeg faktisk vil ha færre og færre å ta gj.snitt av "live". Kanskje det kan vekstes?  
<img src="../images/RNN_avg.png" alt="df"/><br>

### [25.09.2020]
Standarisert slik at alle de predikerte område har spenner over samme område (mean=1, std=.006)  
Nå klarer den å komme med et ganske godt forslag på utrent data. Neste punkt blir å finne ut når dette faktisk er positivt.  
Ser 30 tilbake og forslår 10 fremover.  
<img src="../images/RNN_10_ahead.png" alt="df"/><br>

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