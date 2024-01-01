import { Ai } from '@cloudflare/ai';
import { Hono } from 'hono';
import ui from './ui.html';
import write from './write.html';

const app = new Hono();

app.get("/ui", c => {
	return c.html(ui);
})

app.get("/write", c => {
	return c.html(write);
})

app.post("/notes", async c => {
	const ai = new Ai(c.env.AI);
	const { text } = await c.req.json();
	if (!text) c.throw(400, "No text provided");

	const { results } = await c.env.DB.prepare("INSERT INTO notes (text) VALUES ($1) RETURNING *").bind(text).run();
	const record = results.length ? results[0] : null;
	if (!record) c.throw(500, "Failed to create note");

	const { data } = await ai.run('@cf/baai/bge-base-en-v1.5', { text: [text] });
	const values = data[0];
	if (!values) c.throw(500, "Failed to get embeddings");

	const { id } = record;
	const inserted = await c.env.VECTORIZE_INDEX.upsert([{ id: id.toString(), values }]);
	return c.json({ id, text, inserted });
})


app.get("/", async c => {
	const ai = new Ai(c.env.AI);

	const question = c.req.query('text') || "What is five minus 2?";

	const embeddings = await ai.run('@cf/baai/bge-base-en-v1.5', { text: question });
	const vectors = embeddings.data[0];
	console.log("vectors")
	console.log(vectors)

	const SIMILARITY_CUTOFF = 0.5;
	const vectorQuery = await c.env.VECTORIZE_INDEX.query(vectors, { topK: 1 });
	console.log("vectorQuery")
	console.log(vectorQuery)

	const vecIds = vectorQuery.matches.filter(vec => vec.score > SIMILARITY_CUTOFF).map(vec => vec.id);
	console.log("vecIds")
	console.log(vecIds)

	let notes = []
	if (vecIds.length) {
		const query = `SELECT * FROM notes WHERE id IN (${vecIds.join(", ")})`
		const { results } = await c.env.DB.prepare(query).bind().all();
		if (results) notes = results.map(vec => vec.text)
	}
	console.log("notes")
	console.log(notes)

	const contextMessage = notes.length
		? `Context:\n${notes.map(note => `- ${note}`).join("\n")}`
		: ""

	const systemPrompt = `When answering the question or responding, use the context provided, if it is provided and relevant.`

	console.log("contextMessage")
	console.log(contextMessage);

	const { response: answer } = await ai.run(
		'@cf/meta/llama-2-7b-chat-int8',
		{
			messages: [
				...(notes.length ? [{ role: 'system', content: contextMessage }] : []),
				{ role: 'system', content: systemPrompt },
				{ role: 'user', content: question }
			]
		}
	)

	return c.text(answer)
})

app.onError((err, c) => {
	return c.text(err)
})


export default app;