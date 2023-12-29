
# cose da fare
- Controlli su certe funzioni critiche [in parte fatta]
- Riprendere matrix per weights [da controllare ma fatto, vedere se "buttare" matrix.rs, transpose e map sono veramente necessari?]
- Refactoring and commenting (vedere se si può ottimizzare qualcosa)
- Fare test abbondanti per controllare che tutto funzioni effettivamente bene
- Readme
- Controllare come han fatto gli altri l'anno scorso

# cose carine
- Make logging conditional (Trace, Debug, Error?) [da controllare ma fatto]
- In-Depth logging with multiple log file per experiment [da capire]






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