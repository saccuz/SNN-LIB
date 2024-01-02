
# Cose da fare
- Fare test abbondanti per controllare che tutto funzioni effettivamente bene [Mandare email]
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