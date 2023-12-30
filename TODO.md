
# Cose da fare
- Refactoring and commenting (vedere se si può ottimizzare qualcosa)
- Fare test abbondanti per controllare che tutto funzioni effettivamente bene
- Vogliamo spezzettare snn.rs in layer.rs and snn.rs? (problemi di pub...)
- Ristrutturare lib e mod correttamente (serve davvero lib.rs?)
- Readme [ongoing]

# Cose carine
- Identificare i cambiamenti negli output della rete per ogni iterazione della simulazione (salvare i risultati ?)


<!---
thread_local!(static LOG: Cell<Logging> = Cell::new(Logging::Verbose));

#[derive(Clone, Copy)]
pub enum Logging {
Verbose,
Info,
Debug,
}

pub fn set_logging(level: Logging) {
LOG.replace(level);
}

pub fn get_logging() -> Logging {
LOG.get()
}
-->