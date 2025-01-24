# WebAppTemplate

## 事前準備

### 環境変数設定

#### .env
- サーバーを起動するマシンのIPアドレス
- `.env_tempalte`からコピーする

    ```
    VITE_APP_IP = "XX.XX.XX.XX"
    ```

## スタート

### Docker環境

``` bash
# コンテナ起動
# ★docker-compose.ymlのプロキシ設定のコメントアウトを外しておく
docker compose up -d --build

# バックエンドのコンテナ入る
docker exec -it webapptemp_backend_ct bash

# フロントエンドエンドのコンテナ入る
docker exec -it webapptemp_frontend_ct bash
```

### バックエンド側

``` bash
# バックエンドサーバー起動
python server.py
```

### フロントエンド側

``` bash
# 初回のみ
npm install

# フロントエンドサーバー起動
npm run dev
```

### アクセス
`http://XX.XX.XX.XX:5600`にアクセス


