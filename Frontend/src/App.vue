<script setup>
import NavbarItem from '@/components/NavbarItem.vue'
import FooterItem from './components/FooterItem.vue';
</script>

<template>
  <div class="main-container">
    <NavbarItem />
    <router-view>
    </router-view>
    <FooterItem />
  </div>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200;300;400;500;600;700;800&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: 'Poppins', sans-serif;
}
</style>