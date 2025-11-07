# Import the required modules and classes
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import DateMatcher, MultiDateMatcher
import pyspark.sql.functions as F

if __name__ == "__main__":

    document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
    date = DateMatcher().setInputCols("document").setOutputCol("date").setOutputFormat("yyyy/MM/dd")
    multiDate = MultiDateMatcher().setInputCols("document").setOutputCol("multi_date").setOutputFormat("MM/dd/yy")

    nlpPipeline = Pipeline(stages=[document_assembler, date, multiDate])

    text_list = [
        "See you on next monday.",
        "She was born on 02/03/1966.",
        "The project started yesterday and will finish next year.",
        "She will graduate by July 2023.",
        "She will visit doctor tomorrow and next month again.",
    ]
