. ./.env

curl -X POST http://localhost:8001/documents/upload \
  -F "file=@data/The Curious Journals of Ada Lovelace.pdf"
