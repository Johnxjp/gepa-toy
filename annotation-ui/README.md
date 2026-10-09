# Annotation UI

A small local page for labelling support tickets with a queue, priority, type and tags. Python standard library only.

```sh
python3 server.py              # uses config.json
python3 server.py other.json   # or any other config
```

Then open http://127.0.0.1:8000.

## Config

Paths are relative to the config file. The default config reads the label sheet from the project's `data` folder.

```json
{
  "tickets": "../data/customer_support_tickets/splits/label_sheet.tsv",
  "labels": "../data/customer_support_tickets/labels.json",
  "output": "data/annotations.json",
  "id_field": "id",
  "show_fields": ["subject", "body"],
  "port": 8000
}
```

- `tickets`: tab-separated file with a header row. `id_field` names the ID column; `show_fields` are the columns shown to the labeller.
- `labels`: the options for each dropdown, with keys `queue`, `priority`, `type` and `tags`. Tags allow several choices.
- `output`: where labels are saved, as `{ticket_id: {queue, priority, type, tags}}`. Each save writes the file, so you can stop and resume at any time.

## Tests

```sh
python3 -m unittest
```
