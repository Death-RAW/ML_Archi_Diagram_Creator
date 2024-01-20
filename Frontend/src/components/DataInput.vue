<template>
  <div>
    <div class="input-form">
      <form action="#" class="d-flex flex-column">
        <textarea name="description" id="description" cols="80" rows="20"
          placeholder="Enter Software Description Here.." v-model="description"></textarea>
        <!-- <input type="submit" value="Start Building" class="btn btn-warning" @click="onSubmit"> -->
        <button class="btn btn-warning" @click="onSubmit" type="button" :disabled="submitStatus">
          <span v-if="!submitStatus">Start Building</span>
          <div v-else>
            <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
            Building...
          </div>

        </button>
        <p class="text-shadow text-center mt-2">Will take a few seconds. Hang on tight!</p>
      </form>
    </div>
    <p class="text-center mt-4 text-shadow">By uploading a description you agree to our Terms of Service. This site is
      protected by Captcha and its Privacy Policy and Terms of Service apply.</p>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "DataInput",
  data() {
    return {
      submitStatus: false,
      description: ""
    }
  },
  methods: {
    async onSubmit() {
      console.log("Submitted");
      this.submitStatus = true;

      console.log(this.description);
      await this.submitDescription().finally(() => {
        console.log("Building Completed");
      });
    },
    async submitDescription(){
      let instance = this;

      axios.post('https://eed7-2402-d000-8108-38ce-9e1-2083-87e5-12c0.ngrok-free.app/api/description', {
        description: this.description
      })
      .then(function (response) {

        let diagramData = {
          diagram_name: response.data.diagram_name,
          code_snippet: response.data.code_snippet,
          container_diagram_name: response.data.container_diagram_name,
          container_code_snippet: response.data.container_code_snippet
        }

        localStorage.setItem("diagramData", JSON.stringify(diagramData));
        instance.$emit("diagramData", diagramData);
        instance.$emit("submitStatus", true);
      })
      .catch(function (error) {
        console.log(error);
      });
    }
  }
}
</script>

<style scoped>
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

.text-shadow {
  font-size: .75em;
  color: #555;
}

button {
  padding: 10px !important;
}

/* Button is taking totlay width of the container reduce it */
button[type="button"] {
  width: 30%;
  margin: auto;
}
</style>