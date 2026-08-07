# SG-IA ground-truth QA benchmark

`ground_truth_qa.json` contiene 25 domande in italiano per confrontare le risposte dei backend WIKI e RAG sullo stesso materiale.

## Contenuto

- 20 domande basate su una singola fonte.
- 3 domande che richiedono sintesi tra più fonti.
- 2 domande non rispondibili, utili per misurare astensione e allucinazioni.
- Per ogni domanda rispondibile: risposta di riferimento, punti obbligatori, percorso della fonte, pagina/slide/sezione ed estratto probatorio.

I percorsi in `source_path` sono relativi a `WIKI/backend`, indicato dal campo `corpus_root`.

## Valutazione consigliata

Non usare il confronto letterale completo della risposta. Per ogni caso misurare separatamente:

1. **Correttezza**: percentuale di `required_answer_points` correttamente coperti.
2. **Fedeltà**: assenza di affermazioni non supportate dalle fonti indicate.
3. **Citazioni**: recupero del documento corretto e, quando disponibile, del localizzatore corretto.
4. **Astensione**: per `expected_status = insufficient_knowledge`, il sistema deve dichiarare l'insufficienza delle fonti senza inventare informazioni.
5. **Prestazioni**: latenza, token e costo, registrati separatamente per WIKI e RAG.

Una possibile aggregazione per le domande rispondibili è: 60% correttezza, 25% fedeltà e 15% citazioni. I casi non rispondibili vanno invece valutati come successo/fallimento dell'astensione.

## Aggiornamento del benchmark

Il dataset è riferito al manifest `WIKI/backend/wiki/.ingestion-manifest.json` disponibile il 6 agosto 2026. Se cambiano i documenti grezzi, ricontrollare risposta, punti obbligatori ed evidenze prima di riutilizzarlo.
