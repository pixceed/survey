import React, { useState } from "react";
import Header from "../components/Header";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const Hello = () => {
  const [inputValue, setInputValue] = useState("");
  const [response, setResponse] = useState(null);

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleApiCall = async () => {
    try {
      const res = await fetch(`http://${import.meta.env.VITE_APP_IP}:${import.meta.env.VITE_BACKEND_PORT}/echo`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: inputValue }),
      });
      const data = await res.json();
      setResponse(data);
    } catch (error) {
      console.error("API call failed:", error);
      setResponse({ error: "Failed to fetch data" });
    }
  };

  return (
    <div className="min-h-screen bg-zinc-200">
      <Header />

      <main className="mx-auto px-4 pt-2 h-[calc(100vh-4rem)] flex justify-center">
        <div className="py-3 max-w-7xl w-full flex flex-col gap-4">
          <h1 className="text-2xl font-bold">Hello</h1>

          {/* 入力エリア */}
          <Input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Enter your input"
            className="w-full bg-white"
          />

          {/* API実行ボタン */}
          <Button onClick={handleApiCall} variant="default">
            Call API
          </Button>

          {/* レスポンス表示エリア */}
          {response && (
            <div className="mt-4 p-4 bg-gray-100 border border-gray-300 rounded">
              <h2 className="text-lg font-semibold">API Response:</h2>
              <pre className="mt-2 whitespace-pre-wrap break-all">{JSON.stringify(response, null, 2)}</pre>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Hello;
