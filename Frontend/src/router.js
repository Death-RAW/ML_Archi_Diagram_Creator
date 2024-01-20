import { createRouter, createWebHistory } from 'vue-router'
import Home from './views/Home.vue'
import Generator from './views/Generator.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/generator',
    name: 'Generator',
    component: Generator
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router;