# cose da fare
- ~~pacchettizzare i layer per ogni thread (vd. chat discord)~~
- refactoring 
- rendere operazioni aritmetiche generiche in fault.rs
- fare i controlli che se nella snn non ci sono innerweights (innerconnections) non puoi selezionare inner weights fault 
(in realtà modificare la get_actual_fault perchè deve scegliere casualmente solo fra i layer che hanno true come inner connections)
- fare test abbondanti per controllare che tutto funzioni effettivamente bene



# cose carine
- ottimizzazione
- refactoring
- simHw a parte [mi sa di no]

# Importante
- Cercare di effettuare un log di tutti i fault, quindi visualizzare dove il fault va ad agire e altre informazioni utili per capire cosa sta cambiando nel programma.