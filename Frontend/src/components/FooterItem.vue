<template>
  <div class="container footer-container">
    <div class="d-flex justify-content-between">
      <div>
        <a class="navbar-brand" href="#"><span id="logo-start">THE</span> ARCHITECT</a>
      </div>
      <ul>
        <li class="header"><a href="#">Product</a></li>
        <li><a href="#">Features </a></li>
        <li><a href="#">API</a></li>
        <li><a href="#">Solutions</a></li>
      </ul>
      <ul>
        <li class="header"><a href="#">Company</a></li>
        <li><a href="#">About Us </a></li>
        <li><a href="#">Blog</a></li>
        <li><a href="#">Contact Us</a></li>
      </ul>
      <ul>
        <li class="header"><a href="#">Support</a></li>
        <li><a href="#">Support Center</a></li>
        <li><a href="#">System Status</a></li>
        <li><a href="#">Languages</a></li>
      </ul>
      <ul>
        <li class="header"><a href="#">Follow Us</a></li>
      </ul>
    </div>
    <hr />
    <div class="d-flex justify-content-between">
      <span>&copy; 2022 Yasitha Gamage</span>
      <span>Terms & Conditions | Privacy Policy</span>
    </div>
  </div>
</template>

<script>
export default {
  name: "footer-view"
}
</script>

<style scoped>
.footer-container {
  /* margin: 0 auto; */
  margin-top: 2em;
  padding-bottom: 10px;
}

#logo-start {
  color: #F9C040;
  font-weight: bold;
}

ul {
  list-style: none;
}

ul li a {
  text-decoration: none;
  color: #777;
}

ul > li.header > a{
  color: #000;
  font-weight: 500;
  margin-bottom: 20px;
}

li:not(:last-child) { 
   margin-bottom: 3px;  
}
</style>