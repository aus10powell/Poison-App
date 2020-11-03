mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"powella100@gmail.com git push heroku master\"\n\
" > ~/.streamlit/credentials.toml
echo “\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
“ > ~/.streamlit/config.toml
