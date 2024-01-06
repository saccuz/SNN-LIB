
# Cose da fare
- Fare test
- Readme [ongoing]
- Report (? o basta il readme?)
- presentazione powerpoint????????

# Cose carine
- Identificare i cambiamenti negli output della rete per ogni iterazione della simulazione (salvare i risultati ?)


# DUBBI da discutere
- lif.rs riga 170, incrementiamo fin da subito il t_s_last di time_step => metterlo inizialmente a -time_step?
  (nota che così l'esponente viene 0, e nota anche che quando poi spika viene resettato a 0, e al next forward
    invece sì che è corretto dire che è passato un time_step, solo il primissimo increment forse è da evitare)
- lif.rs il compare tra v_mem e v_th viene fatto fino a tantissime cifre decimali....approssimiamo? (guarda neuron_test Simo)
    la mia calcolatrice ad esempio già tagliava dove ho tagliato io nei test
- layer.rs riga 153 + le altre uguali, alla chiamata di n.forward passiamo il riferimento a tutta la matrice di weights e 
    a tutta la matrice di states weights, ma in realtà basta solo la giusta riga!!! 
    (così evitiamo anche di dover creare la variabile n_neuron in forward di lif.rs)