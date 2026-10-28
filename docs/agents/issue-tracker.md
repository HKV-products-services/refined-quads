# Issue tracker: lokale markdown

Issues en PRD's van deze repo staan als markdownbestanden in `.scratch/issues/` in de
repo-root. `.scratch/` staat in `.gitignore`: issues zijn lokaal en worden niet meegecommit.

## Afspraken

- Eén feature per map: `.scratch/issues/<feature-slug>/`
- De PRD is `.scratch/issues/<feature-slug>/PRD.md`
- Implementatie-issues: `.scratch/issues/<feature-slug>/issues/<NN>-<slug>.md`, genummerd vanaf `01`
- Triagestatus als regel `Status: <label>` bovenaan elk issuebestand (zie `triage-labels.md`)
- Opmerkingen en discussie onderaan het bestand onder een kop `## Comments`
- Termen uit `CONTEXT.md` gebruiken

## Als een skill zegt "publish to the issue tracker"

Maak een nieuw bestand onder `.scratch/issues/<feature-slug>/` (map zo nodig aanmaken).

## Als een skill zegt "fetch the relevant ticket"

Lees het bestand op het opgegeven pad. De gebruiker geeft meestal het pad of het issuenummer.
