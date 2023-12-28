# cose da fare
- ~~pacchettizzare i layer per ogni thread (vd. chat discord)~~
- ~~CAMBIARE COME VENGONO APPLICATI I FAULT NEI VARI COMPONENTI:~~
- ~~1. Ai componenti statici (pesi, threshold, etc...) il fault Stuck at-X deve essere applicato al tempo 0~~
- ~~2. Ai Componenti dinamici (membrana, adder etc...) il fault Stuck at-X deve essere applicato ogni qualvolta il valore cambia.~~
- ~~rendere operazioni aritmetiche generiche in fault.rs~~
~~- fare i controlli che se nella snn non ci sono innerweights (innerconnections) non puoi selezionare inner weights fault 
(in realtà modificare la get_actual_fault perchè deve scegliere casualmente solo fra i layer che hanno true come inner connections)~~
- Aggiungere funzioni utili al logging e controlli su certe funzioni critiche [in parte fatta]
- refactoring and commenting
- fare test abbondanti per controllare che tutto funzioni effettivamente bene


# Logging
- ~~Implement logging~~
- Make logging conditional (Trace, Debug, Error?) [da capire]
- In-Depth logging with multiple log file per experiment [da capire]

# cose carine
- ~~refactoring~~
- ottimizzazione [in parte fatta, da capire]
- simHw a parte [mi sa di no]