<template>
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container">

      <router-link class="navbar-brand" aria-current="page" to="/"><span id="logo-start">THE</span> ARCHITECT</router-link>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNavAltMarkup"
        aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
        <div class="navbar-nav">
          <router-link class="nav-link" aria-current="page" to="/generator">Generate Diagram</router-link>
          <a class="nav-link" href="#">How to use</a>
          <a class="nav-link" href="#">Samples</a>
          <a class="nav-link disabled" href="#" tabindex="-1" aria-disabled="true">Contact</a>
          <div>
            <button class="btn btn-warning mx-3">Log in</button>
            <button class="btn btn-warning">Sign up</button>
          </div>
        </div>
        <!-- <form class="d-flex">
          <input class="form-control me-2" type="search" placeholder="Search" aria-label="Search">
          <button class="btn btn-outline-success" type="submit">Search</button>
        </form> -->
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'NavbarItem'
}
</script>

<style scoped>
.navbar {
  background-color: #fff !important;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  padding: 1.5rem 0;
}

#logo-start {
  color: #F9C040;
  font-weight: bold;
}

#navbarNavAltMarkup {
  flex-grow: 0 !important;
}

button {
  border-radius: 10px;
}
</style>