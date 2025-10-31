# The OpenVoiceOS Technical Manual

![](https://github.com/OpenVoiceOS/ovos_assets/blob/master/Logo/ovos-logo-512.png?raw=true)

the OVOS project documentation is written and maintained by users just like you! 

Think of these docs both as your starting point and also forever changing and incomplete

Please [open Issues and Pull Requests](https://github.com/OpenVoiceOS/ovos-technical-manual)!

User oriented docs are automatically published at https://openvoiceos.github.io/community-docs

Dev oriented docs (this repo) are automatically published at https://openvoiceos.github.io/ovos-technical-manual

## Offline PDF

To generate a single PDF that contains every Markdown document in this
repository, run:

```
python3 scripts/generate_pdf.py
```

The script writes `ovos-technical-manual.pdf` to the repository root. This
file aggregates the project README, the developer documentation under `docs/`
and the localized content under `it/` into one searchable document.