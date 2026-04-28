import { pipeline } from '@huggingface/transformers';

async function analisarSentimento() {
    try {
        console.log("Carregando modelo multilíngue otimizado...");

        // Usamos um modelo que possui suporte nativo ao Transformers.js (formato ONNX)
        const classifier = await pipeline('sentiment-analysis', 'Xenova/bert-base-multilingual-uncased-sentiment');

        const textos = [
            "Este produto é maravilhoso, estou muito feliz!",
            "Horrível, não recomendo a ninguém. Dinheiro jogado fora.",
            "A entrega atrasou um pouco, mas o item é bom."
        ];

        console.log("\n--- Resultados ---\n");

        for (const texto of textos) {
            const resultado = await classifier(texto);
            const res = resultado[0];
            
            // O modelo retorna estrelas (1 star até 5 stars)
            console.log(`Texto: "${texto}"`);
            console.log(`Resultado: ${res.label} (Confiança: ${(res.score * 100).toFixed(2)}%)\n`);
        }
    } catch (error) {
        console.error("Erro ao processar:", error);
    }
}

analisarSentimento();