import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'generation',
      component: () => import('../views/GenerateView.vue')
    },
    {
      path: '/generation',
      name: 'generation',
      component: () => import('../views/GenerateView.vue')
    },
    {
      path: '/display/:sequence',
      name: 'display',
      component: () => import('../views/DisplayView.vue')
    },
    {
      path: '/display',
      name: 'display_none',
      component: () => import('../views/DisplayView.vue')
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue')
    }
  ]
})

export default router