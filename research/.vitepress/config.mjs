import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import mathjax3 from 'markdown-it-mathjax3'

export default withMermaid(
  defineConfig({
    title: 'Deep Learning trong Viễn thám',
    description: 'Nghiên cứu ứng dụng CNN và Deep Learning trong phân tích ảnh viễn thám',
    base: '/sen_doc/',
    lang: 'vi-VN',

    head: [
      ['meta', { name: 'theme-color', content: '#3eaf7c' }],
      ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ],

    themeConfig: {
      nav: [
        { text: 'Trang chủ', link: '/' },
        { text: 'Giới thiệu', link: '/chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning' },
        { text: 'TorchGeo', link: '/chuong-05-torchgeo/muc-01-tong-quan/01-tong-quan' },
        { text: 'xView', link: '/chuong-04-xview-challenges/00-gioi-thieu-xview' },
      ],

      sidebar: [
        {
          text: 'Chương 1: Giới thiệu',
          collapsed: false,
          items: [
            { text: '1.1. Tổng quan CNN & Deep Learning', link: '/chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning' },
          ]
        },
        {
          text: 'Chương 2: Cơ sở lý thuyết',
          collapsed: false,
          items: [
            { text: '2.1.1. Kiến trúc CNN cơ bản', link: '/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban' },
            { text: '2.1.2. Backbone Networks', link: '/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/02-backbone-networks' },
            { text: '2.2.1. Phân loại ảnh', link: '/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/01-phan-loai-anh' },
            { text: '2.2.2. Phát hiện đối tượng', link: '/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/02-phat-hien-doi-tuong' },
            { text: '2.2.3. Phân đoạn ngữ nghĩa', link: '/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/03-phan-doan-ngu-nghia' },
            { text: '2.2.4. Instance Segmentation', link: '/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/04-instance-segmentation' },
          ]
        },
        {
          text: 'Chương 3: Kiến trúc Mô hình',
          collapsed: true,
          items: [
            { text: '3.1. Tổng quan TorchGeo', link: '/chuong-03-kien-truc-model/muc-01-tong-quan/01-tong-quan' },
            { text: '3.2. Classification Models', link: '/chuong-03-kien-truc-model/muc-02-classification/01-classification-models' },
            { text: '3.3. Segmentation Models', link: '/chuong-03-kien-truc-model/muc-03-segmentation/01-segmentation-models' },
            { text: '3.4. Change Detection', link: '/chuong-03-kien-truc-model/muc-04-change-detection/01-change-detection-models' },
            { text: '3.5. Pre-trained Weights', link: '/chuong-03-kien-truc-model/muc-05-pretrained-weights/01-pretrained-weights' },
          ]
        },
        {
          text: 'Chương 4: xView Challenges',
          collapsed: false,
          items: [
            { text: 'Giới thiệu xView', link: '/chuong-04-xview-challenges/00-gioi-thieu-xview' },
            {
              text: '4.1. xView1 - Object Detection',
              collapsed: true,
              items: [
                { text: 'Dataset', link: '/chuong-04-xview-challenges/muc-01-xview1-object-detection/01-dataset' },
                { text: 'Giải nhất', link: '/chuong-04-xview-challenges/muc-01-xview1-object-detection/02-giai-nhat' },
                { text: 'Giải nhì', link: '/chuong-04-xview-challenges/muc-01-xview1-object-detection/03-giai-nhi' },
                { text: 'Giải ba', link: '/chuong-04-xview-challenges/muc-01-xview1-object-detection/04-giai-ba' },
                { text: 'Giải tư', link: '/chuong-04-xview-challenges/muc-01-xview1-object-detection/05-giai-tu' },
                { text: 'Giải năm', link: '/chuong-04-xview-challenges/muc-01-xview1-object-detection/06-giai-nam' },
              ]
            },
            {
              text: '4.2. xView2 - Building Damage',
              collapsed: true,
              items: [
                { text: 'Dataset (xBD)', link: '/chuong-04-xview-challenges/muc-02-xview2-building-damage/01-dataset' },
                { text: 'Giải nhất', link: '/chuong-04-xview-challenges/muc-02-xview2-building-damage/02-giai-nhat' },
                { text: 'Giải nhì', link: '/chuong-04-xview-challenges/muc-02-xview2-building-damage/03-giai-nhi' },
                { text: 'Giải ba', link: '/chuong-04-xview-challenges/muc-02-xview2-building-damage/04-giai-ba' },
                { text: 'Giải tư', link: '/chuong-04-xview-challenges/muc-02-xview2-building-damage/05-giai-tu' },
                { text: 'Giải năm', link: '/chuong-04-xview-challenges/muc-02-xview2-building-damage/06-giai-nam' },
              ]
            },
            {
              text: '4.3. xView3 - Maritime (SAR)',
              collapsed: true,
              items: [
                { text: 'Dataset', link: '/chuong-04-xview-challenges/muc-03-xview3-maritime/01-dataset' },
                { text: 'Giải nhất', link: '/chuong-04-xview-challenges/muc-03-xview3-maritime/02-giai-nhat' },
                { text: 'Giải nhì', link: '/chuong-04-xview-challenges/muc-03-xview3-maritime/03-giai-nhi' },
                { text: 'Giải ba', link: '/chuong-04-xview-challenges/muc-03-xview3-maritime/04-giai-ba' },
                { text: 'Giải tư', link: '/chuong-04-xview-challenges/muc-03-xview3-maritime/05-giai-tu' },
                { text: 'Giải năm', link: '/chuong-04-xview-challenges/muc-03-xview3-maritime/06-giai-nam' },
              ]
            },
          ]
        },
        {
          text: 'Chương 7: Kết luận',
          collapsed: false,
          items: [
            { text: '7.1. Tổng kết', link: '/chuong-07-ket-luan/muc-01-tong-ket/01-ket-luan' },
          ]
        },
      ],

      socialLinks: [
        { icon: 'github', link: 'https://github.com/tchatb/sen_doc' }
      ],

      search: {
        provider: 'local'
      },

      outline: {
        level: [2, 3],
        label: 'Mục lục trang'
      },

      footer: {
        message: 'Nghiên cứu Ứng dụng Deep Learning trong Viễn thám',
        copyright: '2024'
      },

      docFooter: {
        prev: 'Trang trước',
        next: 'Trang sau'
      },

      lastUpdated: {
        text: 'Cập nhật lần cuối'
      },

      returnToTopLabel: 'Về đầu trang',
      sidebarMenuLabel: 'Menu',
      darkModeSwitchLabel: 'Giao diện'
    },

    mermaid: {
      theme: 'default'
    },

    mermaidPlugin: {
      class: 'mermaid'
    },

    markdown: {
      lineNumbers: true,
      config: (md) => {
        md.use(mathjax3)
      }
    },

    vue: {
      template: {
        compilerOptions: {
          isCustomElement: (tag) => tag.startsWith('mjx-')
        }
      }
    }
  })
)
