<template>
  <div class="container card-body">
    <div v-if="!submitStatus">
      <h3 class="text-center mb-5 px-5">Input a Software Description to generate <br /> the Architecture Diagram</h3>
      <DataInput @submitStatus="onSubmitStatus" />
    </div>
    <div class="d-flex flex-column align-items-center" v-else>
      <h3 class="text-center mb-5 px-5">Generated Architecture Diagram</h3>
      <div class="btn-group" role="group" aria-label="Diagram States">
        <button type="button" class="btn btn-warning" @click="onDiagramState('diagram')">Diagram</button>
        <button type="button" class="btn btn-warning" @click="onDiagramState('code')">Code View</button>
      </div>
      <div v-if="diagramState">
        <!-- Context Diagram -->
        <h4 class="mt-5 mb-3">Context Diagram</h4>
        <img :src="imagePath" alt="Asset Image" />
        <a :href="imagePath" download target="_blank" class="btn btn-warning btn-download">Download Diagram</a>

        <!-- Container Diagram -->
        <h4 class="mt-5 mb-3">Container Diagram (Suggested)</h4>
        <img :src="containerImagePath" alt="Asset Image" />
        <a :href="containerImagePath" download target="_blank" class="btn btn-warning btn-download">Download Diagram</a>

      </div>
      <!-- Context Diagram -->
      <div v-else style="width: 100%;">
        <h4 class="mt-5 mb-3">Context Diagram</h4>
        <div class="d-flex flex-column input-form">
          <textarea name="description" id="description" v-model="codeSnippet" cols="80" rows="20"
            placeholder="Architecture Flow.."></textarea>
          <button class="btn btn-warning btn-rebuild" type="button" @click="rebuildDiagram('context')"> Rebuild Diagram</button>
        </div>

        <!-- Container Diagram -->
        <h4 class="mt-5 mb-3">Container Diagram (Suggested)</h4>
        <div class="d-flex flex-column input-form">
          <textarea name="description" id="description" v-model="containerCodeSnippet" cols="80" rows="20"
            placeholder="Architecture Flow.."></textarea>
          <button class="btn btn-warning btn-rebuild" type="button" @click="rebuildDiagram('container')"> Rebuild Diagram</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import DataInput from '../components/DataInput.vue'
import axios from "axios";

export default {
  data() {
    return {
      submitStatus: false,
      diagramState: true,
      imagePath: "images/loading.gif",
      codeSnippet: "",
      containerImagePath: "images/loading.gif",
      containerCodeSnippet: ""
    }
  },
  methods: {
    onSubmitStatus(status) {
      this.submitStatus = status;

      this.updateDiagram();
    },
    onDiagramState(state) {
      switch (state) {
        case "diagram":
          this.diagramState = true;
          break;
        case "code":
          this.diagramState = false;
          break;
        default:
          this.diagramState = true;
      }
    },
    updateDiagram() {
      // Read diagram data from local storage
      let diagramData = JSON.parse(localStorage.getItem("diagramData"));
      console.log("Data", diagramData);

      this.imagePath = "https://eed7-2402-d000-8108-38ce-9e1-2083-87e5-12c0.ngrok-free.app/outputs/" + diagramData.diagram_name;
      this.containerImagePath = "https://eed7-2402-d000-8108-38ce-9e1-2083-87e5-12c0.ngrok-free.app/outputs/" + diagramData.container_diagram_name;
      this.codeSnippet = diagramData.code_snippet;
      this.containerCodeSnippet = diagramData.container_code_snippet;
    },
    rebuildDiagram(type) {
      // Rebuid diagram part
      let rebuildCodeSnippet = "";

      switch (type) {
        case "context":
          rebuildCodeSnippet = this.codeSnippet;
          break;
        case "container":
          rebuildCodeSnippet = this.containerCodeSnippet;
          break;
        default:
          rebuildCodeSnippet = this.codeSnippet;
      }

      axios.post('https://eed7-2402-d000-8108-38ce-9e1-2083-87e5-12c0.ngrok-free.app/api/code', {
        code_snippet: rebuildCodeSnippet
      })
        .then(function (response) {
          let diagramData = {
            diagram_name: response.data.diagram_name,
            code_snippet: response.data.code_snippet
          }

          // Update the diagram data in local storage
          const updatedDiagrams = JSON.parse(localStorage.getItem("updatedDiagrams")) || [];
          updatedDiagrams.push(diagramData);
          localStorage.setItem("updatetDiagrams", JSON.stringify(updatedDiagrams));
          // Download the diagram automatically
          window.open("https://eed7-2402-d000-8108-38ce-9e1-2083-87e5-12c0.ngrok-free.app/outputs/" + diagramData.diagram_name, "_blank");
        })
        .catch(function (error) {
          console.log(error);
        });
    }
  },
  components: { DataInput },
  name: "GeneratorView"
}
</script>

<style scoped>
.card-body {
  margin-top: 4rem;
  margin-bottom: 6rem;
}

.btn-group {
  margin-bottom: 2rem;
}

.input-form {
  background-color: #fff;
  border-radius: 10px;
  padding: 1em;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  margin: 0 1rem;
}

textarea {
  padding: 10px;
  border: none;
  width: inherit;
}

textarea:focus,
textarea:focus-visible {
  border: none !important;
  outline: none;
}

img {
  width: 100%;
}

/* Button is taking totlay width of the container reduce it */
.btn-rebuild {
  width: 30%;
  margin: auto;
}

.btn-download {
  width: 30%;
  /* margin: auto; */
  display: block;
  margin: 0 auto;
}
</style>