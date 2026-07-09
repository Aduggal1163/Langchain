"""
Text splitters and chunking stretergies
Optimizing chunking for RAG 
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter,Language
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

#Sample documents for testing
SAMPLE_TEXT = """
# Baggy Jeans

Baggy jeans have become one of the most popular fashion trends...

# Advantages

One of the biggest advantages of baggy jeans is their comfort...

## Celebrity Influence

Another reason for the popularity of baggy jeans is the influence of celebrities...

### Conclusion

Baggy jeans have successfully re-established themselves as an essential part of contemporary fashion.
"""

SAMPLE_CODE = """
class Solution {
    public int removeCoveredIntervals(int[][] intervals) {
        int cnt=0;
        Arrays.sort(intervals,(a,b)->{
            if (a[0] == b[0]) return b[1] - a[1];
            return a[0] - b[0];
        });
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for(int[] interval : intervals) {
            if(interval[0] >= min && interval[1] <= max) {
                cnt++;
            }
            min = Math.min(min,interval[0]);
            max = Math.max(max,interval[1]);
        }
        return intervals.length-cnt;
    }
}

class Solution {
    public List<List<Integer>> filterOccupiedIntervals(int[][] intervals, int freeStart, int freeEnd) {
        List<int[]> list = new ArrayList<>();
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        for(int[] interval : intervals) {
            if(list.isEmpty() || list.get(list.size()-1)[1] < interval[0]-1) list.add(interval);
            else {
                int start = Math.min(interval[0],list.get(list.size()-1)[0]);
                int end = Math.max(list.get(list.size()-1)[1], interval[1]);
                list.set(list.size()-1,new int[]{start,end});
            }
        }
        List<List<Integer>> adj=new ArrayList<>();
        for(int elt[] : list) {
            int a=elt[0];
            int b=elt[1];
            if(a>freeEnd || b<freeStart) {
                adj.add(Arrays.asList(a,b));
            }
            else {
                if(a < freeStart) {                
                    adj.add(Arrays.asList(a,freeStart-1));
            }
            if(b > freeEnd) {
                adj.add(Arrays.asList(freeEnd+1,b));
            }
        }
        }
        return adj;
    }
}
"""

def recursive_splitter():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators=['\n\n','\n'," ",""]
    )

    chunks = splitter.split_text(SAMPLE_TEXT)
    print("-----RECURSIVE CHARACTER TEXT SPLITTER------")
    print(f"Original Length of document is: {len(SAMPLE_TEXT)}")
    print(f"Number of chunk(s): {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    print(f"\n First chunk preview:\n {chunks[0][:200]}")

def overlap_importance():

    text = "The quick brown fox jumps over a lazy dog."*10
    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap = 0
    )
    splitter_with_overlap = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap = 20
    )
    chunks_no_overlap = splitter_no_overlap.split_text(text)
    chunks_with_overlap = splitter_with_overlap.split_text(text)
    print("#----------Overlap importance---------")
    print("Without overlap")
    print(f"Chunk one end: {chunks_no_overlap[0][-20:]}")
    print(f"Chunk two start: {chunks_no_overlap[1][:20]}")
    print("---------")
    print("With overlap")
    print(f"Chunk one end: {chunks_with_overlap[0][-20:]}")
    print(f"Chunk two start: {chunks_with_overlap[1][:20]}")

def markdown_splitter():

    headers_to_consider = [
        ('#','h1'),
        ('##','h2'),
        ('###','h3')
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)

    chunks = splitter.split_text(SAMPLE_TEXT)
    print("#----------Markdown splitter---------")
    print(f"Markdown text splitter produced {len(chunks)} chunks.")
    for i,chunk in enumerate(chunks):
        print(f"Metadata is: {chunk.metadata}\n")
        print(f"Content is : {chunk.page_content[:200]}\n")

def code_splitter():

    splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size = 500,
        chunk_overlap = 50,
        language=Language.JAVA
        )

    chunks = splitter.split_text(SAMPLE_CODE)
    print("#----------Code splitter---------")
    print(f"No of chunks produced are: {len(chunks)}")
    for i,chunk in enumerate(chunks):
        print(f"\n Chunk {i+1} which has {len(chunk)} character(s). Code is: \n {chunk[:200]}")

def document_splitter():
    loader = PyPDFLoader("langchain.pdf")
    document = loader.load()
    print(f"Loading {len(document)} documents from DPF")

    splitter = RecursiveCharacterTextSplitter(
         chunk_size = 500,
         chunk_overlap=50
     )
    chunks = splitter.split_documents(document)
    print("#----------Document splitter---------")
    print(f"Splitted into {len(chunks)} document(s)")
    print(f"First chunk metadata: {chunks[0].metadata}")
    print(f"First chunk ending: {chunks[0].page_content[-50:]}")
    print(f"Second chunk starting: {chunks[0].page_content[:80]}")

# def exercise():
def exercise():
    loader = PyPDFLoader('langchain.pdf')
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 30,
    )
    document = splitter.split_documents(docs)
    model = init_chat_model(
        model='gpt-4o-mini',
        model_provider='openai',
        temperature=0.7
    )
    summary_prompt = ChatPromptTemplate.from_template("Generate me one line summary of: {text}")
    category_prompt = ChatPromptTemplate.from_template("Also choose the category from the following options Artificial Intelligence,Machine Learning, Deep Learning,Python,Other {text}")
    analyzed_chain = RunnableParallel(
        summary = summary_prompt | model | StrOutputParser(),
        category = category_prompt | model | StrOutputParser()
    )
        
    print(f"\nTotal number of chunks are: {len(document)}")
    for i,doc in enumerate(document,start=1):
        print("="*50,f"chunk {i}","="*50)
        print(f"\ncontent is: {doc.page_content}\nmetadata is: {doc.metadata}")
        response = analyzed_chain.invoke({'text':doc.page_content})
        print(f"\n summary for this chunks is: {response['summary']}")
        print(f"\nCategory for this chunk is: {response['category']}")

if __name__ == '__main__':
    # recursive_splitter()
    # overlap_importance()
    # markdown_splitter()
    # code_splitter()
    # document_splitter()
    exercise()