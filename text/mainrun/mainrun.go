package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	dlp "github.com/bytedance/godlp"
	dlpheader "github.com/bytedance/godlp/dlpheader"
)

type DLPHandler struct {
	engine dlpheader.EngineAPI
}

// 初始化脱敏处理器
func NewDLPHandler(caller string) (*DLPHandler, error) {
	eng, err := dlp.NewEngine(caller)
	if err != nil {
		return nil, fmt.Errorf("初始化失败: %v", err)
	}

	// 加载默认配置
	if err := eng.ApplyConfigDefault(); err != nil {
		return nil, fmt.Errorf("加载配置失败: %v", err)
	}

	return &DLPHandler{engine: eng}, nil
}

// 注册自定义脱敏规则
func (h *DLPHandler) RegisterCustomRules() {
	// 邮箱保留域名全称
	h.engine.RegisterMasker("KeepDomain", func(email string) (string, error) {
		parts := strings.Split(email, "@")
		if len(parts) != 2 {
			return email, fmt.Errorf("invalid email format")
		}
		return fmt.Sprintf("***%s@%s", parts[0][len(parts[0])-1:], parts[1]), nil
	})
}

// 处理文本内容
func (h *DLPHandler) ProcessText(content string) (string, error) {
	// 先检测敏感信息
	results, err := h.engine.Detect(content)
	if err != nil {
		return "", fmt.Errorf("检测失败: %v", err)
	}
	log.Printf("检测到%d条敏感信息", len(results))

	// 执行脱敏处理
	masked, _, err := h.engine.Deidentify(content)
	if err != nil {
		return "", fmt.Errorf("脱敏失败: %v", err)
	}
	return masked, nil
}

// HTTP处理函数
func handleTextAPI(handler *DLPHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 设置CORS头
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Content-Type", "application/json")

		// 处理预检请求
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// 只处理POST请求
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
			return
		}

		// 解析请求体
		var request struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "invalid request body"})
			return
		}

		// 处理文本
		masked, err := handler.ProcessText(request.Text)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}

		// 返回结果
		response := map[string]string{"result": masked}
		json.NewEncoder(w).Encode(response)
	}
}

func main() {
	// 初始化处理器
	handler, err := NewDLPHandler("data.processor.v1")
	if err != nil {
		log.Fatal(err)
	}
	defer handler.engine.Close()

	// 注册自定义规则
	handler.RegisterCustomRules()

	// 设置HTTP路由
	http.HandleFunc("/api/text", handleTextAPI(handler))

	// 启动服务器
	log.Println("启动服务，监听端口 :8081")
	log.Fatal(http.ListenAndServe(":8081", nil))
}
