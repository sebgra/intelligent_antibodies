<script setup>
defineProps({
  msg: {
    type: String,
    required: true
  }
})
</script>

<template>
  <div class="wrapper">
    <img alt="IAView" class="logo" src="@/assets/logo.svg" width="125" height="125" />
    <h1 class="title">{{ msg }}</h1>
  </div>
</template>

<style scoped>
.wrapper {
  display: flex;
  justify-content: space-between;
  border: lightgrey;
  border-radius: 12px;
}
.title {
  font-weight: 500;
  font-size: 2.6rem;
  margin: auto;
  margin-left: 0;
  position: relative;
  color: #38a3a5;
}

.logo {
  display: block;
  margin: auto;
  margin-right: 1rem;
  fill: #22577a;
}

.greetings h1,
.greetings h3 {
  text-align: center;
}
</style>
