## Human evaluation result
We manually evaluated 1000 triplets, which are top-5 predictions of 200 contexts from WikiZSL. *human_eval_wikiZSL.tsv* contains each triplet and labels from WikiZSL and by our manual evaluation.

### Columns 
* input_text: Context containing triplet(s)	
* gold_head: Gold head entity originally annotated by WikiZSL  	
* gold_tail: Gold tail entity originally annotated by WikiZSL  	
* gold_rel: Gold relation originally annotated by WikiZSL  	
* pred_head: Head entity pedicted by ZETT
* pred_tail: Tail entity pedicted by ZETT
* pred_rel:	Relation predicted by ZETT
* Label from Wiki-ZSL
* Label from our human evaluation