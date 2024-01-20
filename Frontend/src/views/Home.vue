<template>
  <div>
    <div class="container-fluid">
      <div class="container">
        <div class="d-flex flex-column justify-content-center content">
          <p>Generate <span class="highlight">Software Architecture</span><br> Diagram</p>
          <div>
            <a class="btn btn-warning mr-4" role="button">Get Started</a>
            <a class="btn btn-warning" role="button">Preview</a>
          </div>
        </div>
      </div>
    </div>
    <div class="container">
      <div class="d-flex flex-column align-content-center">
        <h4 class="text-center mb-4 title">Generate Architecture Diagram</h4>
        <DataInput />
      </div>
    </div>
    <div class="container">
      <div class="d-flex align-items-center content-slide">
        <div class="left-text">
          <h4 class="mb-5">Generate Architecture Diagrams automatically in seconds with one click.</h4>
          <p>Thanks to Architects clever AI, you can generate C4 model architecture diagrams slashing desgin time.</p>
        </div>
        <img src="images/asset-one.png" alt="Asset Image">
      </div>
    </div>
    <div class="container">
      <div class="d-flex flex-column align-content-center slider-bg mb-4">
        <h4 class="text-center mb-4 title">Stunning Solutions</h4>
        <img src="images/sample-diagram.png" alt="Asset Image">
        <p>EG is an energy provider who allows their customers to access the service using web browsers and also from
          mobile devices. The online service will allow users to login to the system and submit the electricity and gas
          meter readings. EG will notify users on successful submission of meter readings also sends the bill amount based
          on the calculation.</p>
      </div>
    </div>
  </div>
</template>

<script>
import DataInput from '../components/DataInput.vue';
import Generator from './Generator.vue';

export default {
  name: "HomeView",
  components: { DataInput, Generator }
}
</script>

<style scoped>
.container-fluid {
  background: url('/images/bg-img.png') no-repeat right;
}

.container {
  padding: 2em;
  /* height: 60vh; */
}

.content {
  font-size: 1.9em;
  color: #000;
  height: inherit;
  margin: auto 0;
  height: 85vh;
}

.highlight {
  font-weight: 600;
}

.btn {
  margin-right: 10px;
}

/* Header Bottom Border */
.title::after {
  content: "";
  display: block;
  width: 15%;
  height: 3px;
  background-color: #FFC92C;
  margin: 10px auto;
}

.content-slide>img {
  width: 30%;
  margin-left: 5em;
  padding: 5em 0;
}

.left-text {
  /* width: 40%; */
  padding: 10%;
  text-align: justify;
}

.left-text h4 {
  font-size: 2em;
}

.slider-bg {
  background: url('/images/bg-img-two.png') no-repeat center;
}

.slider-bg img {
  margin: 2em auto;
  width: 70%;
  padding: 20px;
  border: 2px solid #55555553;
  border-radius: 10px;
}

.slider-bg p {
  margin: 20px;
  padding: 0 4rem;
  text-align: center;
  color: #717171;
}

</style>