# Domeindocumentatie

Hoe de engineering-skills de domeindocumentatie van deze repo gebruiken bij het
verkennen van de code. Deze repo is **single-context**.

## Lees dit vóór het verkennen

- **`CONTEXT.md`** in de root: de begrippen (refined quad mesh, base cell, refinement
  level, lattice, transition rows, transition cell, footprint, replacement map, domain,
  refinement area, water area, water removal rule).

Ontbreekt dit bestand, **ga dan stil door**. Meld het ontbreken niet en stel niet voor
ze vooraf aan te maken. De producerende skill (`/grill-with-docs`) maakt ze pas aan wanneer
begrippen of beslissingen echt vastgelegd worden.

## Bestandsindeling

```
/
├── CONTEXT.md
├── docs/
│   └── agents/
├── src/refined_quads/
└── tests/
```

## Gebruik de woorden uit de woordenlijst

Noem je een domeinbegrip (in een issuetitel, refactorvoorstel, hypothese of testnaam),
gebruik dan de term zoals `CONTEXT.md` die definieert. Wijk niet uit naar synoniemen die
de woordenlijst juist vermijdt.

Staat het begrip er nog niet in, dan is dat een signaal: je verzint taal die het project
niet gebruikt (heroverweeg dat), of er is een echt gat (noteer het voor `/grill-with-docs`).
