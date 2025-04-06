. ./.env

curl -X POST \
     -H "Content-Type: application/json" \
     -d '{
         "query": "Who is the father of Ada , Countess of Lovelace?",
         "top_k": 3
     }' \
     http://127.0.0.1:8001/search
