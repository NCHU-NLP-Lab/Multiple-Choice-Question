const button = document.getElementById("btn-submit");
const resultBox = document.getElementById("results-div");
const resultBox_outside = document.getElementById("results-outside")

const spinner = document.createElement("span");
spinner.classList.add("spinner-grow");
spinner.classList.add("spinner-grow-sm");
spinner.setAttribute("role", "status");
spinner.setAttribute("aria-hidden", true);

function toggleButtonState() {
  if (button.disabled) {
    button.disabled = false;
    button.removeChild(spinner);
    button.innerText = "選擇題問題生成";
  } else {
    button.innerText = "問題生成中";
    button.insertBefore(spinner, button.firstChild);
    button.disabled = true;
  }
}

function toggleResultBox(show = true) {
  if (show === true) {
    resultBox.style.display = "block";
    resultBox_outside.style.backgroundColor="rgb(41, 105, 176, 0.7)";
  } else {
    resultBox.style.display = "none";
    resultBox_outside.style.backgroundColor="rgb(0, 0, 0, 0)";
  }
}

function successResultBox() {
  resultBox.classList.remove("alert-danger");
  resultBox.classList.add("alert-success");
}

function failResultBox() {
  resultBox.classList.remove("alert-success");
  resultBox.classList.add("alert-danger");
}

function fillTextGenResult(modelOutput) {
  var output = "";
  var number_of_data = modelOutput.length;
  for(i = 0; i < number_of_data ; i ++)
  {
    if(i%2==0)
    {
      output += modelOutput[i]+"<br>";
    }
    else if(i%2==1 && i!=number_of_data-1)
    {
      output += modelOutput[i]+"<dvi><br><br><br></div>";
    }
    else
    {
      //最後一行不空白
      output += modelOutput[i];
    }
  }
  console.log(output)
  resultBox.innerHTML = output;
}

async function submitForm(event) {
  event.preventDefault();
  //輸入從html輸入BOX拿取
  const maintext = document.getElementById("sentance-1").value;

  console.log(maintext);
  toggleButtonState();
  toggleResultBox(false);
  try {
    //fetch() 回傳的 promise 不會 reject HTTP 的 error status，就算是 HTTP 404 或 500 也一樣
    await fetch("/mcq", {
      //對應前面app.py的@app.post("/zhqa")
      method: "POST",
      // JSON.stringify將 JavaScript 值轉換為 JSON 字串
      body: JSON.stringify({maintext}),
      //指定內容為 JSON 格式，以 UTF-8 字符編碼進行編碼。
      headers: new Headers({
        "Content-Type": "application/json; charset=UTF-8",
      }),
    })
      .then(async (response) => {
        if (!response.ok) {
          const errorDetail = JSON.stringify(await response.json());
          throw new Error(
            `Request failed for ${response.statusText} (${response.status}): ${errorDetail}`
          );
        }
        return response.json();
      })//成功取得輸入(data)
      .then((data) => {
        console.log(data);
        successResultBox();
        fillTextGenResult(data);
      });
  } catch (error) {
    console.error(error);
    failResultBox();
    resultBox.innerText = error;
  } finally {
    toggleButtonState();
    toggleResultBox();
  }
}
//找尋答案之btn
button.addEventListener("click", submitForm, false);
